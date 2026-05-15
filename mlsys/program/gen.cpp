#include <cstdlib>
#include <string>
#include "json.hpp"
using json = nlohmann::json;
int main() {
    system("g++ -o generator generator.cpp \"-Wl,--stack=1024000000\" -std=c++17");
    for (int i = 0; i < 10; i++) {
        std::string command = "generator.exe example" + std::to_string(i) + ".json";
        system(command.c_str());
    }
    return 0;
}