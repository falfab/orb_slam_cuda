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
#pragma once

#include <fstream>
#include <thread>
#include <mutex>

#include <stdint.h>
#include <sys/termios.h>
#include <time.h>
#include <stdio.h>

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>

#include "Timer.hpp"

class Powermon {

static const int VERSION_H = 2;
static const int VERSION_L = 2;
static const int OVF_FLAG = 7;
static const int TIME_FLAG = 6;
static const int DONE_FLAG = 5;

#define BAUDRATE B1000000
static const double V_FULLSCALE;
static const double R_SENSE;    // found empirically
static const double I_FULLSCALE;

public:

    /**
     * @brief Create the Powermoon object.
     * @details
     *
     * @param p Name of the device port ( e.g. /dev/ttyUSB0 ).
     * @param m Bit Mask of sensors, every 1 bit enable a sensor.
    */
    Powermon();
    ~Powermon();
    Powermon(std::ofstream* _log);
    Powermon(const char* _device, uint16_t _mask);
    Powermon(const char* _device, uint16_t _mask, std::ofstream* _log);
    Powermon(const char* _device, uint16_t _mask, std::ofstream* _log, int _sampling_interval);

    /**
     * @brief Start getting samples from the powermoon.
     * @details Start getting samples from the powermoon, it's a blocking function, it waits that powermoon starts sending samples before return.
     *
     * @return 1 if something got wrong
     */
    void start();
    /**
     * @brief Stops getting samples if it is running.
     * @details Stops getting samples if it is running.
     */
    void stop();
    /**
     * @brief Waits the end of getting samples execution.
     * @details Waits the end of getting samples execution.
     */
    void wait();
    
    void pauseCollection();
    void resumeCollection();
    
    void resetData();
    void resetDataBlocking();
    bool hasToResetData();
    

    /**
     * @brief Sets the output stream where logs will be written.
     * @details Sets the output stream where logs will be written.
     *
     * @param out Log output stream.
     * @param divider String that divides every information in a line.
     * @param time Logs with time?
     */
    //void log(std::ostream* out,bool time);

    double getPowerMax();
    double getPowerAvg();
    double getPowerTotal();
    int getSampledInstants();
	
	void prepare();
	void hardStop();
	void startAsync();
	void stopAsync();
	void readSync();
	void resetDataCollected();

    void printStats();



private:
    //const int N_OF_SAMPLING = 30000;
	//8160 massimi sample che danno valore corretto!
	const int N_OF_SAMPLING = 8160;
    //const int N_OF_SAMPLING = 0;

    int sampling_interval;
    int active_sensors;

    double power_max;
    double power_total;
    int sampled_instants;
    bool powermon_found = false;
    /**
     * @brief Configure the tty, connect and open it
     * @return 0 all ok, !=0 if there is some errors
     */
    bool configure();

    void Reset(bool err = true);
    /**
     * @brief Set sensors mask to powermoon
     * @return 0 all ok, !=0 if there is some errors
     */
    void setMask();
    /**
     * @brief Set the number of samples to powermoon
     * @return 0 all ok, !=0 if there is some errors
     */
    void setSamples();
    /**
     * @brief Set the current time to powermoon
     * @return 0 all ok, !=0 if there is some errors
     */
    void setTime();
    /**
     * @brief Reset the powermoon
     * @details Reset the powermoon, mutex necessary if thread is running
     * @return 0 all ok, !=0 if there is some errors
     */
    void reset();

    void getVersion();

    //PARAMS
    const char*    device;
    const unsigned short mask;
    const bool log_enabled;
    std::ostream*    log_stream;

    //THREAD
    std::mutex        mutex;
    std::thread*   supportThread;
    static void    task(void* ptr);
    bool            stop_task;
    volatile bool           first_read;
    volatile bool           should_read;
    volatile bool           should_reset;

    //SERIAL
    int         pw_file_descriptor;
    FILE*         pw_file_pointer;
    
    char* dummy_char;

    //CONFIG
    struct termios pw_config, pw_old_config;

    timer::Timer<timer::HOST> t;
};
