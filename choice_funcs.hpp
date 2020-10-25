#ifndef CHOICE_FUNCS_HPP
#define CHOICE_FUNCS_HPP


void add_instance_size(InstanceSize& instance_size,
        const Eigen::Matrix<double, -1, -1>& m) {
    instance_size.push_back(m.rows());
    instance_size.push_back(m.cols());
}

void add_instance_size(InstanceSize& instance_size, const int i) {
    instance_size.push_back(i);
}

InstanceSize default_instance_size_func() {
    return InstanceSize();
}

template <typename T, typename... Args>
InstanceSize default_instance_size_func(const T& arg, const Args&... args) {
    auto instance_size = default_instance_size_func(args...);
    add_instance_size(instance_size, arg);
    return instance_size;
}

template <typename R, typename... Args>
std::function<R(Strategy&, InstanceSize, Args...)> get_choice_func(
        std::array<std::function<R(Args...)>, DEVICE_COUNT> funcs) {
    return [funcs](Strategy& strategy, InstanceSize instance_size, Args... args) {
        auto choice = strategy.choose(instance_size);

        auto start = Clock::now();
        auto result = funcs[choice](args...);
        auto end = Clock::now();

        strategy.update(instance_size, choice, end - start);
        return result;
    };
}

template <typename R, typename... Args>
std::function<R(Args...)> get_simple_choice_func(
        std::function<R(Strategy&, InstanceSize, Args...)> choice_func,
        Strategy& strategy,
        std::function<InstanceSize(Args...)> instance_size_func
            = default_instance_size_func<Args...>) {
    return [choice_func, &strategy, instance_size_func](const Args&&... args) {
        return choice_func(strategy, instance_size_func(args...), args...);
    };
}

template <typename R, typename... Args>
class SmartFunction {
private:
    std::array<std::function<R(Args...)>, DEVICE_COUNT> funcs;
public:
    SmartFunction(std::array<std::function<R(Args...)>, DEVICE_COUNT> funcs) : funcs(funcs) {};
    R operator()(Strategy& strategy, InstanceSize instance_size, Args... args) {
        auto choice = strategy.choose(instance_size);

        auto start = Clock::now();
        auto result = funcs[choice](args...);
        auto end = Clock::now();

        strategy.update(instance_size, choice, end - start);
        //std::cout << choice << ": " << to_string(instance_size) << to_seconds(end - start) << "\n";

        return result;
    }
};

template <typename R, typename... Args>
class SimpleSmartFunction {
private:
    SmartFunction<R, Args...> func;
    Strategy& strategy;
    std::function<InstanceSize(Args...)> size_func;
public:
    SimpleSmartFunction(SmartFunction<R, Args...> func, Strategy& strategy, std::function<InstanceSize(Args...)> size_func)
        : func(func), strategy(strategy), size_func(size_func) {};
    R operator()(Args... args) {
        return multiply_func(strategy, size_func(args...), args...);
    }
};





#endif