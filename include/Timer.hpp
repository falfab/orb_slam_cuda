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
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#pragma once

#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
//#include "Base/Host/fUtil.hpp"

//#define __COLOR

#if defined(__COLOR)
    #include "fUtil.hpp"
    #define __ENABLE_COLOR(...)  __VA_ARGS__
#else
    #define __ENABLE_COLOR(...)
#endif

#if defined(__linux__)
	#include <ctime>		//CPU
	#include <sys/times.h>	//SYS
    #include <unistd.h>
#endif

namespace timer {

enum timerType {  HOST = 0					// Wall (real) clock host time
			#if defined(__linux__)
				, CPU  = 1				// User time
				, SYS  = 2				// User/Kernel/System time
			#endif
};

/**
* @class Timer
* @brief Timer class for HOST and DEVICE
* HOST timer: "HOST" (default) Wall (real) clock host time, "CPU" User time, "SYS" User/Kernel/System time
* "DEVICE" Wall clock device time
*/
template<timerType type>
class Timer {

protected:
	int decimals;
	int space;
    __ENABLE_COLOR(StreamModifier::Color defaultColor;)
	std::ostream& outStream = std::cout;
private:
	// HOST
	std::chrono::steady_clock::time_point startTime, endTime;
#if defined(__linux__)
	// CPU
	std::clock_t c_start, c_end;
	// SYS
	struct tms startTMS, endTMS;
#endif

public:
	/**
	* Default costructor
	*/
#if defined(__COLOR)
    Timer(int _decimals = 1, int _space = 15,
          StreamModifier::Color color = StreamModifier::FG_DEFAULT);
#else
    Timer(int _decimals = 1, int _space = 15);
#endif
	Timer(std::ostream& _outStream, int _decimals = 1);
    virtual ~Timer();

	/** Start the timer */
	virtual void start();

	/** Stop the timer */
	virtual void stop();

	/*
	* Get the time elapsed between start() and stop()
	* @return time elapsed
	*/
	template<typename _ChronoPrecision = std::chrono::duration<float,
                                                               std::milli>>
	float duration();
	
	template<typename _ChronoPrecision = std::chrono::duration<uint64_t,
                                                               std::nano>>
	uint64_t durationNano();

	/*
	* Print the time elapsed between start() and stop()
	* if start() and stop() not invoked indef behavior
	*/
	template<typename _ChronoPrecision = std::chrono::duration<float,
                                                               std::milli>>
	void print(std::string str);

	/*
	* Stop the timer and print the time elapsed between start() and stop()
	* if start() and stop() not invoked indef behavior
	*/
	template<typename _ChronoPrecision = std::chrono::duration<float,
             std::milli>>
	void getTime(std::string str);

    template<typename _ChronoPrecision = std::chrono::duration<float,
         std::milli>>
    void getTimeA(std::string str);
};

} //@xlib

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
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include <stdexcept>
#include <iostream>
#include <iomanip>				// set precision cout
#include <chrono>
#include <ratio>

namespace std {

template<class Rep, std::intmax_t Num, std::intmax_t Denom>
std::ostream& operator<<(std::ostream& os,
                         __attribute__((unused)) const std::chrono::duration
                         <Rep, std::ratio<Num, Denom>>& ratio) {

    if (Num == 3600 && Denom == 1)		return os << " h";
    else if (Num == 60 && Denom == 1)	return os << " min";
    else if (Num == 1 && Denom == 1)	return os << " s";
    else if (Num == 1 && Denom == 1000)	return os << " ms";
    else return os << " Unsupported";
}

} //@std

namespace timer {

using float_seconds = std::chrono::duration<float>;

template<typename>
struct is_duration : std::false_type {};

template<typename T, typename R>
struct is_duration<std::chrono::duration<T, R>> : std::true_type {};

//-------------------------- GENERIC -------------------------------------------

template<timerType type>
Timer<type>::Timer(std::ostream& _outStream, int _decimals) :
             outStream(_outStream), decimals(_decimals) {}

#if defined(__COLOR)

template<timerType type>
Timer<type>::Timer(int _decimals, int _space, StreamModifier::Color _color) :
 				   decimals(_decimals), space(_space), defaultColor(_color),
                   startTime(), endTime()
#if __linux__
                    ,c_start(0), c_end(0), startTMS(), endTMS()
#endif
                    {}

#else

template<timerType type>
Timer<type>::Timer(int _decimals, int _space) :
 				   decimals(_decimals), space(_space) {}

#endif

template<timerType type>
Timer<type>::~Timer() {}

template<timerType type>
template<typename _ChronoPrecision>
void Timer<type>::print(std::string str) {
	static_assert(is_duration<_ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
	std::cout __ENABLE_COLOR(<< this->defaultColor)
              << std::right << std::setw(this->space - 2) << str << "  "
			  << std::fixed << std::setprecision(this->decimals)
              << this->duration<_ChronoPrecision>()
			  << _ChronoPrecision()
              __ENABLE_COLOR(<< StreamModifier::FG_DEFAULT)
              << std::endl;
}

template<timerType type>
template<typename _ChronoPrecision>
void Timer<type>::getTime(std::string str) {
	static_assert(is_duration<_ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
	this->stop();
	this->print(str);
}

template<timerType type>
template<typename _ChronoPrecision>
void Timer<type>::getTimeA(std::string str) {
    getTime(str);
    std::cout << std::endl;
}

//-------------------------- HOST ----------------------------------------------

template<>
template<typename _ChronoPrecision>
float Timer<timer::HOST>::duration() {
	static_assert(is_duration<_ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
	return std::chrono::duration_cast<_ChronoPrecision>
                                     (endTime - startTime).count();
}
template<>
template<typename _ChronoPrecision>
uint64_t Timer<timer::HOST>::durationNano() {
	static_assert(is_duration<_ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
	return std::chrono::duration_cast<_ChronoPrecision>
                                     (endTime - startTime).count();
}


//-------------------------- CPU -----------------------------------------------
#if defined(__linux__)

template<>
template<typename _ChronoPrecision>
float Timer<CPU>::duration() {
	static_assert(is_duration<_ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
	return std::chrono::duration_cast<_ChronoPrecision>(
            float_seconds((float) (c_end - c_start) / CLOCKS_PER_SEC) ).count();
}

//-------------------------- SYS -----------------------------------------------

template<>
template<typename _ChronoPrecision>
float Timer<SYS>::duration() {
	throw std::runtime_error( "Timer<SYS>::duration() is unsupported" );
}

template<>
template<typename _ChronoPrecision>
void Timer<SYS>::print(std::string str) {
	static_assert(is_duration<_ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
	auto wall_time = std::chrono::duration_cast<_ChronoPrecision>(
                                                  endTime - startTime ).count();
	auto user_time = std::chrono::duration_cast<_ChronoPrecision>(
                float_seconds( (float) (endTMS.tms_utime - startTMS.tms_utime) /
                ::sysconf(_SC_CLK_TCK) ) ).count();
	auto sys_time = std::chrono::duration_cast<_ChronoPrecision>(
                float_seconds( (float) (endTMS.tms_stime - startTMS.tms_stime) /
                ::sysconf(_SC_CLK_TCK) ) ).count();

	std::cout __ENABLE_COLOR(<< defaultColor)
              << std::setw(space) << str
			  << "  Elapsed time: [user " << user_time << ", system "
              << sys_time << ", real "
			  << wall_time << " " << _ChronoPrecision() << "]"
              __ENABLE_COLOR(<< StreamModifier::FG_DEFAULT)
              << std::endl;
}
#endif
} //@xlib
