/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato, Michele Scala
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include <exception>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string.h>
#include <limits>
#include <numeric>
#include <cmath>
#include <fstream>

#include "powermon.hpp"

const double Powermon::V_FULLSCALE = 26.52;
const double Powermon::R_SENSE = 0.00422;    // found empirically
const double Powermon::I_FULLSCALE = 0.10584;

Powermon::Powermon() : device("/dev/ttyUSB0"),  mask(128u), log_enabled(false), log_stream(NULL),
sampling_interval(1) {
    active_sensors = __builtin_popcount (mask);
}

Powermon::Powermon(std::ofstream* _log) :
                        device("/dev/ttyUSB0"),  mask(128u), log_enabled(true), log_stream(_log),
sampling_interval(1) {
    active_sensors = __builtin_popcount (mask);
}

Powermon::Powermon(const char* _device, uint16_t _mask) :
                        device(_device),  mask(_mask), log_enabled(false), log_stream(NULL),
sampling_interval(1) {
    active_sensors = __builtin_popcount (mask);
}

Powermon::Powermon(const char* _device, uint16_t _mask, std::ofstream* _log) :
                        device(_device),  mask(_mask), log_enabled(true), log_stream(_log),sampling_interval(1) {
                            active_sensors = __builtin_popcount (mask);
                        }

Powermon::Powermon(const char* _device, uint16_t _mask, std::ofstream* _log, int _si) :
                        device(_device),  mask(_mask), log_enabled(true), log_stream(_log), sampling_interval(_si) {
                            active_sensors = __builtin_popcount (mask);
                        }

Powermon::~Powermon() {}

//------------------------------------------------------------------------------

bool Powermon::configure() {
    this->pw_file_descriptor = ::open(device, O_RDWR | O_NOCTTY);
    if (this->pw_file_descriptor < 0)
        //throw std::runtime_error("Powermon::Configure : TTY not found");
        return 0;
    if ((this->pw_file_pointer = ::fdopen(pw_file_descriptor, "r+")) == NULL)
        throw std::runtime_error("Powermon::Configure : Can't open TTY");

    ::tcgetattr(this->pw_file_descriptor, &pw_old_config); /* save current port settings */
    ::bzero(&pw_config, sizeof (pw_config));

    /*
    CRTSCTS == hardware flow control
    CS8 = 8 bits
    CLOCAL = ignore modem control lines
    CREAD = enable receiver
    B38400 = baud rate
    */
    /* configure port */
    this->pw_config.c_iflag = IGNPAR;
    this->pw_config.c_cflag = BAUDRATE | CS8 | CSTOPB | CLOCAL | CREAD;
    this->pw_config.c_oflag = 0;

    /* set input mode (non-canonical, no echo,...) */
    this->pw_config.c_lflag = 0;

    this->pw_config.c_cc[VTIME] = 0; /* inter-character timer unused */
    this->pw_config.c_cc[VMIN] = 1; /* blocking read until 1 char received */

    ::tcflush(this->pw_file_descriptor, TCIFLUSH);
    ::tcsetattr(this->pw_file_descriptor, TCSANOW, &pw_config);
    return 1;
}

void Powermon::prepare()
{
	if (this->configure()) {
        this->powermon_found = true;
        this->power_max = 0;
        this->power_total = 0;
        this->sampled_instants = 0;
    
		this->Reset();
        this->setMask();
        this->setSamples();
        this->setTime();
    } else {
        this->power_max = NAN;
        this->power_total = NAN;
        this->sampled_instants = static_cast<int>(NAN);
        std::cerr << "Not able to connect to Powermon" << std::endl;
    }
}
int errIdx = 0;
void Powermon::startAsync()
{
    if (!powermon_found) return;
	::fprintf(this->pw_file_pointer, "e\n");
	errIdx++;
    if(::fflush(this->pw_file_pointer) != 0)
		std::cerr << "Error flushing start async " << errIdx << std::endl;
}
void Powermon::stopAsync()
{
    if (!powermon_found) return;
	::fprintf(this->pw_file_pointer, "d\n");
    if(::fflush(this->pw_file_pointer) != 0)
		std::cerr << "Error flushing stop async" << std::endl;
	std::this_thread::sleep_for(std::chrono::milliseconds(5));
}
void Powermon::resetDataCollected()
{
	this->power_max = 0;
	this->power_total = 0;
	this->sampled_instants = 0;
}
void Powermon::hardStop()
{
	if (!powermon_found)
        return;
    
    this->Reset(false);
    ::tcflush(this->pw_file_descriptor, TCIFLUSH);
    ::tcsetattr(this->pw_file_descriptor, TCSANOW, &pw_old_config);
    ::close(this->pw_file_descriptor);
}
	
void Powermon::Reset(bool err) {
    char buffer[32];

    ::fprintf(this->pw_file_pointer, "d\n");
    ::usleep(100000);
    ::fflush(this->pw_file_pointer);
    ::usleep(100000);
    ::tcflush(this->pw_file_descriptor, TCIFLUSH);
    ::fprintf(this->pw_file_pointer, "\n");
    //get result
    dummy_char = ::fgets(buffer, sizeof(buffer), this->pw_file_pointer);
    if (err && ::strcmp(buffer, "OK\r\n") != 0) {
        std::cerr << "-> " << buffer << std::endl;
        throw std::runtime_error("Powermon::Reset");
    }
}

void Powermon::setMask() {
    char buffer[32];

    unsigned length = ::sprintf(buffer, "m %u\n", this->mask);
    ::fwrite(buffer, 1, length, this->pw_file_pointer);
    //get response
    dummy_char = ::fgets(buffer, sizeof(buffer), this->pw_file_pointer);

    unsigned setval;
    ::sscanf(buffer, "M=%u\r", &setval);
    //get result
    dummy_char = ::fgets(buffer, sizeof(buffer), this->pw_file_pointer);
    if (::strcmp(buffer, "OK\r\n") != 0)
        throw std::runtime_error("Powermon::setMask");
}

void Powermon::setSamples() {
    char buffer[32];

    unsigned length = ::sprintf(buffer, "s %u %u\n", sampling_interval, N_OF_SAMPLING);
    ::fwrite(buffer, 1, length, this->pw_file_pointer);
    dummy_char = ::fgets(buffer, sizeof(buffer), this->pw_file_pointer);

    unsigned set_interval, set_num_samples;
    ::sscanf(buffer, "S=%u,%u\r", &set_interval, &set_num_samples);

    dummy_char = ::fgets(buffer, sizeof(buffer), this->pw_file_pointer);
    if (::strcmp(buffer, "OK\r\n") != 0)
        throw std::runtime_error("Powermon::setSamples");
}

void Powermon::setTime() {
    char buffer[32];

    unsigned length = ::sprintf(buffer, "t %u\n", 1);
    ::fwrite(buffer, 1, length, this->pw_file_pointer);
    dummy_char = ::fgets(buffer, sizeof(buffer), this->pw_file_pointer);

    unsigned setval;
    ::sscanf(buffer, "T=%u\r", &setval);

    dummy_char = ::fgets(buffer, sizeof(buffer), this->pw_file_pointer);
    if (::strcmp(buffer, "OK\r\n") != 0)
        throw std::runtime_error("Powermon::setTime");
}

//------------------------------------------------------------------------------

void Powermon::getVersion() {
    char buffer[128];
    int length;

    length = sprintf(buffer, "v\n");
    fwrite(buffer, 1, length, pw_file_pointer);

    //get response
    dummy_char = fgets(buffer, sizeof(buffer), pw_file_pointer);
    std::cout << "Version: " << buffer << std::endl;

    dummy_char = fgets(buffer, sizeof(buffer), pw_file_pointer);
    std::cout << buffer << std::endl;

    dummy_char = fgets(buffer, sizeof(buffer), pw_file_pointer);
    std::cout << buffer << std::endl;

    //get result
    dummy_char = fgets(buffer, sizeof(buffer), pw_file_pointer);
    if (::strcmp(buffer, "OK\r\n") != 0)
        throw std::runtime_error("Powermon::getVersion");
}

//------------------------------------------------------------------------------


void Powermon::start() {
    if (this->configure()) {
        this->powermon_found = true;
        this->power_max = 0;
        this->power_total = 0;
        this->sampled_instants = 0;

        this->stop_task = false;
        this->first_read = false;
        this->should_read = false;
        this->Reset();
        this->setMask();
        this->setSamples();
        this->setTime();

        supportThread = new std::thread(Powermon::task, this);
        while(!first_read);
    } else {
        this->power_max = NAN;
        this->power_total = NAN;
        this->sampled_instants = static_cast<int>(NAN);
        std::cerr << "Not able to connect to Powermon" << std::endl;
    }
}

void Powermon::stop() {
    if (!powermon_found)
        return;
    stop_task = true;
    (*supportThread).join();
    delete supportThread;

    this->Reset(false);
    ::tcflush(this->pw_file_descriptor, TCIFLUSH);
    ::tcsetattr(this->pw_file_descriptor, TCSANOW, &pw_old_config);
    ::close(this->pw_file_descriptor);
}

int Powermon::getSampledInstants() {
    return this->sampled_instants;
}

double Powermon::getPowerMax() {
    return this->power_max;
}

double Powermon::getPowerAvg() {
    return (double) this->power_total / this->sampled_instants;
}

double Powermon::getPowerTotal() {
    return this->power_total;
}

void Powermon::pauseCollection() {
		this->should_read = false;
}
void Powermon::resumeCollection() {
		this->should_read = true;
}

void Powermon::resetData() {
		this->should_reset = true;
}
void Powermon::resetDataBlocking() {
		this->should_reset = true;
		while(this->should_reset);
}
bool Powermon::hasToResetData() {
		return this->should_reset;
}

//------------------------------------------------------------------------------

void Powermon::task(void* ptr){
		size_t dummy_size;
    Powermon* master = (Powermon*) ptr;
    master->mutex.lock();

    ::fprintf(master->pw_file_pointer, "e\n");

    int times[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    std::stringstream ss_mod3;
    int sensor_index = 0;
    double power_sensor[master->active_sensors];

    while (true) {
        unsigned char buffer[4];
        dummy_size = ::fread(buffer, 1, 4, master->pw_file_pointer);
        uint8_t flags = buffer[0];

        if (!(flags & (1<<TIME_FLAG)) && (flags & (1<<DONE_FLAG))) break;
        if(flags & (1 << OVF_FLAG))
            std::cerr << "*" << std::endl;
        if (flags & (1 << TIME_FLAG)) {
            unsigned int time = (((uint32_t)buffer[0] << 24) | ((uint32_t)buffer[1] << 16)
                                | ((uint32_t)buffer[2] << 8) | ((uint32_t)buffer[3]))
                                & 0x3FFFFFFF;
            //std::cerr << "timestamp: " << time <<std::endl;
        } else {
            master->first_read = true;
            
            //std::cout << master->should_reset << std::endl << std::flush;
            if(master->should_reset)
            {
            	master->power_max = 0;
							master->power_total = 0;
							master->sampled_instants = 0;
            	master->should_reset = false;
            }
            if(!master->should_read) continue;

            uint8_t sensor = (uint8_t) buffer[0] & 0x0F;
            uint16_t voltage = ((uint16_t) buffer[1] << 4) | ((buffer[3] >> 4) & 0x0F);
            uint16_t current = ((uint16_t) buffer[2] << 4) | (buffer[3] & 0x0F);
            double v_double = (double) voltage / 4096 * V_FULLSCALE;
            double i_double = (double) current / 4096 * I_FULLSCALE / R_SENSE;

            double watt = v_double * i_double;
            power_sensor[sensor_index] = watt;

            if (master->log_enabled) {
                ss_mod3 << times[sensor] << "\t" << (int) sensor
                                         << "\t" << v_double << "\t" << i_double
                                         << "\t" << watt << std::endl;
            }
            times[sensor]++;

            sensor_index++;
            if (sensor_index == master->active_sensors) {
                sensor_index = 0;
                if (master->log_enabled){
                    (*(master->log_stream)) << ss_mod3.rdbuf();}

                double sum = std::accumulate(power_sensor, power_sensor + master->active_sensors, 0);
                master->power_total += sum;
                if (sum > master->power_max)
                    master->power_max = sum;

                master->sampled_instants++;
            }

            //STOP READ ACTIVE_SENSORS
            if (master->stop_task) break;
        }
    }
    
    /*master->Reset();
    ::tcflush(master->pw_file_descriptor, TCIOFLUSH);
    ::tcsetattr(master->pw_file_descriptor, TCSANOW, &master->pw_old_config);
    ::close(master->pw_file_descriptor);*/

    master->mutex.unlock();
}




void Powermon::readSync()
{
    if (!powermon_found) return;

	size_t dummy_size;
		
	int times[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    std::stringstream ss_mod3;
    int sensor_index = 0;
    double power_sensor[active_sensors];

	int idx = 0;
	long byte_read = 0;
	bool data_end = false;
	const int SZ_BUFFER = 8160;
			
	bool ovf_flag = false;
    while (true) {
        unsigned char buffer[4*SZ_BUFFER];
		
		fd_set set;
		FD_ZERO(&set); /* clear the set */
		int filedesc = fileno(pw_file_pointer);
		FD_SET(filedesc, &set); /* add our file descriptor to the set */
		struct timeval timeout;
		timeout.tv_sec = 0;
		timeout.tv_usec = 0;
		int rv = select(filedesc + 1, &set, NULL, NULL, &timeout);
		
		if(rv == -1)
			std::cerr << "ERROR" << std::endl;/* an error accured */
		else if(rv == 0)
			dummy_size = 0; /* a timeout occured */
		else
			//dummy_size = ::fread(buffer, 1, 4*SZ_BUFFER, pw_file_pointer); /* there was data to read */
			dummy_size = ::read(filedesc, buffer, 4*SZ_BUFFER); /* there was data to read */
		if((dummy_size) & 3 != 0)
		{
			dummy_size += ::fread(buffer+dummy_size, 1, 4-((dummy_size) & 3), pw_file_pointer);
		}
		
		byte_read += dummy_size;
		
		if(dummy_size == 0 && !data_end)
		{
			data_end = true;
			break;
		}
		
		//std::cout << ++idx << " - " << dummy_size << " - " << byte_read << std::endl;
		for( int base_i = 0; base_i < dummy_size; base_i+=4)
        {
			if(buffer[base_i] == 'O' && buffer[base_i+1] == 'K' && buffer[base_i+2] == '\r' && buffer[base_i+3] == '\n')
			{
				break;
			}
				
			uint8_t flags = buffer[base_i + 0];

			if (!(flags & (1<<TIME_FLAG)) && (flags & (1<<DONE_FLAG))) break;
			if(flags & (1 << OVF_FLAG))
			{ 
				if(! ovf_flag) ;//std::cerr << "*" << std::endl;
				ovf_flag = true;
			}
			if (flags & (1 << TIME_FLAG)) {
				unsigned int time = (((uint32_t)buffer[base_i + 0] << 24) | ((uint32_t)buffer[base_i + 1] << 16)
									| ((uint32_t)buffer[base_i + 2] << 8) | ((uint32_t)buffer[base_i + 3]))
									& 0x3FFFFFFF;
				//std::cerr << "timestamp: " << time <<std::endl;
			} else {
				if(    (buffer[base_i + 2] == 32 && buffer[base_i + 3] == 0)
					|| (buffer[base_i + 1] == 32 && buffer[base_i + 2] == 0)
					|| (buffer[base_i + 0] == 32 && buffer[base_i + 1] == 0) 
          || buffer[base_i + 3] == 0 )
					{
						continue; // non valid data?? Probably because interrupted by stop command
					}
				
				uint8_t sensor = (uint8_t) buffer[base_i + 0] & 0x0F;
				uint16_t voltage = ((uint16_t) buffer[base_i + 1] << 4) | ((buffer[base_i + 3] >> 4) & 0x0F);
				uint16_t current = ((uint16_t) buffer[base_i + 2] << 4) | (buffer[base_i + 3] & 0x0F);
				double v_double = (((double) voltage) * V_FULLSCALE ) / 4096;
				double i_double = (((double) current) * I_FULLSCALE ) / ( 4096.0 * R_SENSE );

				double watt = v_double * i_double;
        ///std::cout << watt << "W" << std::endl;
				power_sensor[sensor_index] = watt;

				if (log_enabled) {
					ss_mod3 << times[sensor] << "\t" << (int) sensor
											 << "\t" << v_double << "\t" << i_double
											 << "\t" << watt << std::endl;
				}
				times[sensor]++;

				sensor_index++;
				if (sensor_index == active_sensors) {
					sensor_index = 0;
					if (log_enabled){
						(*(log_stream)) << ss_mod3.rdbuf();}

					double sum = 0;//std::accumulate(power_sensor, power_sensor + active_sensors, 0);
          for(int i = 0; i < active_sensors; i++) sum += power_sensor[i];
					power_total += sum;
					if (sum > power_max)
						power_max = sum;

					sampled_instants++;
				}

				//STOP READ ACTIVE_SENSORS
				//if (master->stop_task) break;
				//if(b) std::this_thread::sleep_for(std::chrono::seconds(10));
			}
        }
		if(power_max > 20)
		{
			std::ofstream myfile;
			myfile.open ("sensors.txt",  std::ios::out | std::ios::app );
			for( int base_i = 0; base_i < dummy_size; base_i+=4)
			{
				uint8_t sensor = (uint8_t) buffer[base_i + 0] & 0x0F;
				uint16_t voltage = ((uint16_t) buffer[base_i + 1] << 4) | ((buffer[base_i + 3] >> 4) & 0x0F);
				uint16_t current = ((uint16_t) buffer[base_i + 2] << 4) | (buffer[base_i + 3] & 0x0F);
				double v_double = (double) voltage / 4096 * V_FULLSCALE;
				double i_double = (double) current / 4096 * I_FULLSCALE / R_SENSE;

				double watt = v_double * i_double;
				
				myfile << (int) buffer[base_i+0] << ";";
				myfile << (int) buffer[base_i+1] << ";";
				myfile << (int) buffer[base_i+2] << ";";
				myfile << (int) buffer[base_i+3] << ";";
				myfile << (int) sensor << ";";
				myfile << (int) voltage << ";";
				myfile << (int) current << ";";
				myfile << (double) v_double << ";";
				myfile << (double) i_double << ";";
				myfile << (double) watt << std::endl;
			}
			myfile.close();
		}
    }
}


void Powermon::printStats()
{
    std::cout << "Power avg         : " << getPowerAvg()        << " W" << std::endl;
    std::cout << "Power max         : " << getPowerMax()        << " W" << std::endl;
    std::cout << "Power total       : " << getPowerTotal()      << " J" << std::endl;
    std::cout << "Sampled instatnts : " << getSampledInstants() << " instants" << std::endl;
    std::cout << "Time              : " << t.duration()         << " ms" << std::endl;
}
