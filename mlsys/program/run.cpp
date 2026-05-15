#include <windows.h>
#include <iostream>
#include <string>

bool run_with_timeout(const std::string& cmd, DWORD timeout_ms) {
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // CreateProcess 需要一个可写的字符串缓冲区
    char* command_line = _strdup(cmd.c_str());

    // 1. 启动进程
    if (!CreateProcessA(
        NULL,           // 模块名
        command_line,   // 命令行参数
        NULL,           // 进程安全属性
        NULL,           // 线程安全属性
        FALSE,          // 句柄继承
        0,              // 创建标志
        NULL,           // 环境变量
        NULL,           // 当前目录
        &si,            // 启动信息
        &pi             // 进程信息
    )) {
        std::cerr << "无法启动进程，错误代码: " << GetLastError() << std::endl;
        free(command_line);
        return false;
    }

    // 2. 等待进程结束，设置超时时间
    // WaitForSingleObject 是核心，它会阻塞直到进程结束或超时
    DWORD wait_result = WaitForSingleObject(pi.hProcess, timeout_ms);

    bool success = false;
    if (wait_result == WAIT_OBJECT_0) {
        // 进程在规定时间内正常结束
        DWORD exit_code;
        GetExitCodeProcess(pi.hProcess, &exit_code);
        success = true;
    } 
    else if (wait_result == WAIT_TIMEOUT) {
        // 3. 发生超时，强制掐断
        TerminateProcess(pi.hProcess, 1); // 1 是给子进程指定的退出码
    }

    // 4. 清理资源
    free(command_line);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return success;
}

int main() {
    // 示例：调用你的 mlsys.exe，限时 5000 毫秒（5秒）
    std::string command = "mlsys.exe benchmarks/mlsys-2026-1.json benchmarks/mlsys-2026-1-out.json";
    
    run_with_timeout(command, 2000); 
    
    return 0;
}