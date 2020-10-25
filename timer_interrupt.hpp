#ifndef TIMER_INTERRUPT_HPP
#define TIMER_INTERRUPT_HPP

#include <vector>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/writer.hpp>

#include "structures.hpp"

class timer_interrupt : public stan::callbacks::interrupt {
private:
    std::vector<TimeType> times_;
public:
    timer_interrupt() {}

    void operator()() { times_.push_back(Clock::now()); }

    auto times() { return times_; }

    auto times(size_t ix) { return times_[ix]; }
};

class timer_writer : public stan::callbacks::writer {
private:
    std::vector<TimeType> times_;
public:
    timer_writer() {}

    void operator()(const std::vector<double>& state) { times_.push_back(Clock::now()); }

    auto times() { return times_; }

    auto times(size_t ix) { return times_[ix]; }
};

#endif