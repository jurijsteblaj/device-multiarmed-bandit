#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include <chrono>
#include <iostream>
#include <vector>
#include <boost/container_hash/hash.hpp>

using InstanceSize = std::vector<size_t>;
using FeatureVector = std::vector<double>;
using Clock = std::chrono::high_resolution_clock;
using Duration = Clock::duration;
using TimeType = std::chrono::time_point<Clock>;

template <typename T>
std::string to_string(std::vector<T> instance_size) {
    std::ostringstream output;
    for (auto el : instance_size) {
        output << el << ", ";
    }
    return output.str();
}

namespace std {

  template <>
  struct hash<std::vector<size_t>>
  {
    std::size_t operator()(const std::vector<size_t>& val) const
    {
      return boost::hash_range(val.begin(), val.end());
    }
  };

}

template <
    class Rep,
    class Period = std::ratio<1>
>
double to_seconds(std::chrono::duration<Rep,Period> duration) {
    return std::chrono::duration<double>(duration).count();
}

std::ostream& operator<<(std::ostream& out, const InstanceSize instance_size) {
    for (auto i = 0; i < instance_size.size(); i++) {
        if (i != 0) {
            out << ",";
        }
        out << instance_size[i];
    }
    return out;
}

std::istream& operator>>(std::istream& in, InstanceSize& instance_size) {
    char ch;
    for (auto i = 0; i < instance_size.size(); i++) {
        if (i != 0) {
            in >> ch;
        }
        in >> instance_size[i];
    }
    return in;
}

std::ostream& operator<<(std::ostream& out, const Duration duration) {
    return out << std::chrono::duration<double>(duration).count();
}

std::istream& operator>>(std::istream& in, Duration& duration) {
    double sec;
    in >> sec;
    duration = std::chrono::duration_cast<Duration>(std::chrono::duration<double>(sec));
    return in;
}


/*
Values are explicit to allow easier access in data structures.
If there is a data structure which efficiently maps all
values of an enum to some other values, that might be a good replacement.
Size of Experiment::_processing_time
(and maybe other structures) needs to match this.
*/
enum Device {
    CPU = 0, GPU,
    DEVICE_COUNT
};

#endif