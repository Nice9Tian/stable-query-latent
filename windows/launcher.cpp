// Root launcher for the distributable package (pattern from CodeX2Thirdpart):
// a tiny static exe at dist/ root that starts resources/LariceAnchorStudioApp.exe
// with the resources/ directory as the working directory, forwarding argv.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

namespace {
constexpr DWORD kPathBufferSize = 32768;

bool fileExists(const wchar_t *path)
{
    const DWORD attributes = GetFileAttributesW(path);
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

bool copyLauncherDirectory(wchar_t *buffer, DWORD bufferSize)
{
    const DWORD length = GetModuleFileNameW(nullptr, buffer, bufferSize);
    if (length == 0 || length >= bufferSize) {
        return false;
    }

    for (DWORD i = length; i > 0; --i) {
        if (buffer[i - 1] == L'\\' || buffer[i - 1] == L'/') {
            buffer[i - 1] = L'\0';
            return true;
        }
    }
    return false;
}

bool joinPath(const wchar_t *base, const wchar_t *relative, wchar_t *output, DWORD outputSize)
{
    const int written = wsprintfW(output, L"%s\\%s", base, relative);
    return written > 0 && static_cast<DWORD>(written) < outputSize;
}

bool copyApplicationPath(const wchar_t *baseDir, wchar_t *appPath, DWORD appPathSize)
{
    if (!joinPath(baseDir, L"resources\\LariceAnchorStudioApp.exe", appPath, appPathSize)) {
        return false;
    }
    if (fileExists(appPath)) {
        return true;
    }

    if (!joinPath(baseDir, L"LariceAnchorStudioApp.exe", appPath, appPathSize)) {
        return false;
    }
    return true;
}

void copyParentDirectory(const wchar_t *path, wchar_t *parent, DWORD parentSize)
{
    lstrcpynW(parent, path, parentSize);
    const int length = lstrlenW(parent);
    for (int i = length; i > 0; --i) {
        if (parent[i - 1] == L'\\' || parent[i - 1] == L'/') {
            parent[i - 1] = L'\0';
            return;
        }
    }
}

DWORD quotedLength(const wchar_t *value)
{
    DWORD length = 2;
    DWORD backslashCount = 0;
    for (const wchar_t *cursor = value; *cursor; ++cursor) {
        if (*cursor == L'\\') {
            ++backslashCount;
            continue;
        }
        if (*cursor == L'"') {
            length += backslashCount * 2 + 2;
            backslashCount = 0;
            continue;
        }
        length += backslashCount + 1;
        backslashCount = 0;
    }
    return length + backslashCount * 2 + 1;
}

void appendQuoted(wchar_t *&cursor, const wchar_t *value)
{
    *cursor++ = L'"';
    DWORD backslashCount = 0;
    for (const wchar_t *source = value; *source; ++source) {
        if (*source == L'\\') {
            ++backslashCount;
            continue;
        }
        if (*source == L'"') {
            for (DWORD i = 0; i < backslashCount * 2 + 1; ++i) {
                *cursor++ = L'\\';
            }
            *cursor++ = *source;
            backslashCount = 0;
            continue;
        }
        for (DWORD i = 0; i < backslashCount; ++i) {
            *cursor++ = L'\\';
        }
        *cursor++ = *source;
        backslashCount = 0;
    }
    for (DWORD i = 0; i < backslashCount * 2; ++i) {
        *cursor++ = L'\\';
    }
    *cursor++ = L'"';
}

wchar_t *buildCommandLine(const wchar_t *appPath)
{
    int argc = 0;
    LPWSTR *argv = CommandLineToArgvW(GetCommandLineW(), &argc);

    DWORD length = quotedLength(appPath);
    for (int i = 1; argv && i < argc; ++i) {
        length += 1 + quotedLength(argv[i]);
    }

    wchar_t *commandLine = static_cast<wchar_t *>(HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, length * sizeof(wchar_t)));
    if (!commandLine) {
        if (argv) {
            LocalFree(argv);
        }
        return nullptr;
    }

    wchar_t *cursor = commandLine;
    appendQuoted(cursor, appPath);
    for (int i = 1; argv && i < argc; ++i) {
        *cursor++ = L' ';
        appendQuoted(cursor, argv[i]);
    }
    *cursor = L'\0';

    if (argv) {
        LocalFree(argv);
    }
    return commandLine;
}

void showError(const wchar_t *message, const wchar_t *path = nullptr)
{
    wchar_t buffer[kPathBufferSize];
    if (path) {
        wsprintfW(buffer, L"%s\n%s", message, path);
        MessageBoxW(nullptr, buffer, L"Larice Anchor Studio", MB_ICONERROR | MB_OK);
        return;
    }
    MessageBoxW(nullptr, message, L"Larice Anchor Studio", MB_ICONERROR | MB_OK);
}
}

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
    wchar_t baseDir[kPathBufferSize];
    wchar_t appPath[kPathBufferSize];
    wchar_t appDir[kPathBufferSize];

    if (!copyLauncherDirectory(baseDir, kPathBufferSize)) {
        showError(L"Could not locate the launcher directory.");
        return 1;
    }

    if (!copyApplicationPath(baseDir, appPath, kPathBufferSize) || !fileExists(appPath)) {
        showError(L"Could not find the application:", appPath);
        return 1;
    }

    copyParentDirectory(appPath, appDir, kPathBufferSize);
    wchar_t *commandLine = buildCommandLine(appPath);
    if (!commandLine) {
        showError(L"Could not build the application command line.");
        return 1;
    }

    STARTUPINFOW startupInfo{};
    startupInfo.cb = sizeof(startupInfo);
    PROCESS_INFORMATION processInfo{};

    const BOOL started = CreateProcessW(
        appPath,
        commandLine,
        nullptr,
        nullptr,
        FALSE,
        0,
        nullptr,
        appDir,
        &startupInfo,
        &processInfo);

    HeapFree(GetProcessHeap(), 0, commandLine);

    if (!started) {
        showError(L"Could not start the application:", appPath);
        return 1;
    }

    CloseHandle(processInfo.hThread);
    CloseHandle(processInfo.hProcess);
    return 0;
}
#else
int main()
{
    return 0;
}
#endif
