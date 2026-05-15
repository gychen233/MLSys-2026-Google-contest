#include <bits/stdc++.h>
using namespace std;
int main() {
    system("g++ -o mlsys mlsys.cpp \"-Wl,--stack=1024000000\" -std=c++17");
    system("mlsys.exe benchmarks/mlsys-1.json benchmarks/mlsys-1-out.json");
    system("mlsys.exe benchmarks/mlsys-2.json benchmarks/mlsys-2-out.json");
    system("mlsys.exe benchmarks/mlsys-3.json benchmarks/mlsys-3-out.json");
    system("mlsys.exe benchmarks/mlsys-4.json benchmarks/mlsys-4-out.json");
    system("mlsys.exe benchmarks/mlsys-5.json benchmarks/mlsys-5-out.json");
    system("mlsys.exe benchmarks/mlsys-2026-1.json benchmarks/mlsys-2026-1-out.json");
    system("mlsys.exe benchmarks/mlsys-2026-5.json benchmarks/mlsys-2026-5-out.json");
    system("mlsys.exe benchmarks/mlsys-2026-9.json benchmarks/mlsys-2026-9-out.json");
    system("mlsys.exe benchmarks/mlsys-2026-13.json benchmarks/mlsys-2026-13-out.json");
    system("mlsys.exe benchmarks/mlsys-2026-17.json benchmarks/mlsys-2026-17-out.json");
    return 0;
}