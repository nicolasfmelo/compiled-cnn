#include "runtime.h"
#include <iostream>
int main() {
    if (engine::sigmoid(-100.f) < 0 || !std::isfinite(engine::silu(-100.f)) || !std::signbit(engine::silu(-1000.f))) return 1;
    std::cout << "PASS scalar activation extremes\n";
}
