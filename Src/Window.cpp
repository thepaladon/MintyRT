#include "Window.h"

#include <cassert>
#include <map>
#include <vector_types.h>
#include <Windows.h>
#include <Windowsx.h>
//----------------------------------------


LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

Window::Window(uint32_t width, uint32_t height, std::string name)
{

    // Initialize the window class.
    m_WindowData.m_Width = width;
    m_WindowData.m_Height = height;
    m_WindowData.m_Alive = true;
    m_WindowData.m_WindowName = name;

    // Converting from string 
    const LPCSTR lName = name.c_str();
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, lName, NULL };
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    RegisterClassEx(&wc);

    m_WindowHandle = CreateWindow(
        wc.lpszClassName,
        lName,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        m_WindowData.m_Width,
        m_WindowData.m_Height,
        nullptr,
        nullptr,
        wc.hInstance,
        NULL);

    // Register for raw input from the mouse
    RAWINPUTDEVICE rawInputDevice;
    rawInputDevice.usUsagePage = 0x01; // Generic desktop controls
    rawInputDevice.usUsage = 0x02;     // Mouse
    rawInputDevice.hwndTarget = m_WindowHandle;
    rawInputDevice.dwFlags = 0;

    if (RegisterRawInputDevices(&rawInputDevice, 1, sizeof(RAWINPUTDEVICE)) == FALSE) {
        printf("Failed lol");
        assert(false);
    }


    ShowWindow(m_WindowHandle, SW_SHOWDEFAULT);

    // Here I pass in struct with data to alter when needed
    SetWindowLongPtr(m_WindowHandle, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&m_WindowData));

    UpdateWindow(m_WindowHandle);

    m_Bitmap = CreateSampleDIB();


}

HBITMAP Window::CreateSampleDIB()
{
    // Check if an existing DIB exists
    if (m_Bitmap) 
    	DeleteObject(m_Bitmap);

    BITMAPINFO bmi = { 0 };
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = this->GetAlignedWidth();
    bmi.bmiHeader.biHeight = -this->GetAlignedHeight();  // Negative height for top-down DIB
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = sizeof(uchar3) * 8; // 24 - bit RGB

    void* pPixels = nullptr;
    const HBITMAP hDIB = CreateDIBSection(nullptr, &bmi, DIB_RGB_COLORS, &pPixels, nullptr, 0);

    m_BitmapInfo = bmi;
    m_Bitmap = hDIB;
    return hDIB;
}





bool Window::OnUpdate(float dt)
{
    // Reset from last frame
    m_WindowData.m_Resized = false;
    m_WindowData.m_MouseDeltaX = 0;
    m_WindowData.m_MouseDeltaY = 0;

    std::string name = m_WindowData.m_WindowName + " || " + std::to_string(dt) + " ms.";
    SetWindowText(m_WindowHandle, name.c_str());

    //  Goes through all messages and events (to process input)
    MSG msg;
    while (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return m_WindowData.m_Alive;
}

void Window::RenderFb(void* fb)
{
    HDC hdc = GetDC(m_WindowHandle);

    // Use SetDIBitsToDevice to copy the DIB to the window's DC
    HDC hMemDC = CreateCompatibleDC(hdc);
    HGDIOBJ hOldBitmap = SelectObject(hMemDC, m_Bitmap);

    int result = SetDIBitsToDevice(
        hdc,  // Destination DC
        0,    // xDest
        0,    // yDest
        m_WindowData.m_Width,  // Width
        m_WindowData.m_Height,  // Height
        0,    // xSrc
        0,    // ySrc
        0,    // uStartScan
        m_WindowData.m_Height,  // cScanLines
        fb,  // Pointer to DIB pixels
        &m_BitmapInfo, // Pointer to BITMAPINFO
        DIB_RGB_COLORS  // Color table usage
    );

    SelectObject(hMemDC, hOldBitmap);
    DeleteDC(hMemDC);

    // Handle errors in result
    if (result == GDI_ERROR) {
        // Handle error here
        printf("Error %i \n", result);
        assert(false);
    }
}

void Window::Shutdown()
{
    m_WindowHandle = nullptr;
}


// Define a map to associate key codes with key names
std::map<int, std::wstring> keyMap = {
    {VK_LEFT, L"Left Arrow"},
    {VK_RIGHT, L"Right Arrow"},
    {VK_UP, L"Up Arrow"},
    {VK_DOWN, L"Down Arrow"},
    {VK_SPACE, L"Space"},
    {VK_RETURN, L"Enter"}
};



// Dispatches messages or events
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    // Getting window to set variables
    WindowData* data = (WindowData*)GetWindowLongPtr(hWnd, GWLP_USERDATA);

    if (!data)
        return DefWindowProc(hWnd, msg, wParam, lParam);

    // Switch between msg
    switch (msg) {
    case WM_ERASEBKGND:
        return 1;
    case WM_DESTROY:
        data->m_Alive = false;

        return 0;
    case WM_SIZE:
    {
        // Get the updated size.
        RECT r;
        GetClientRect(hWnd, &r);

        uint32_t width = r.right - r.left;
        uint32_t height = r.bottom - r.top;

        // Set all data
        data->m_Width = width;
        data->m_Height = height;
        data->m_Resized = true;
        printf("Resizing : %i : %i \n", width, height);

    } break;

    case WM_INPUT: {
        RAWINPUT rawInput;
        UINT size = sizeof(RAWINPUT);
        GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, &rawInput, &size, sizeof(RAWINPUTHEADER));

        if (rawInput.header.dwType == RIM_TYPEMOUSE) {

            data->m_MouseDeltaX = data->m_MouseGlobalPosX - rawInput.data.mouse.lLastX;
            data->m_MouseDeltaY = data->m_MouseGlobalPosY - rawInput.data.mouse.lLastY;

            data->m_MouseGlobalPosX = rawInput.data.mouse.lLastX;
            data->m_MouseGlobalPosY = rawInput.data.mouse.lLastY;
        }

    } break;
    case WM_KEYDOWN:
    {
        // Handle key press events
        const int key = static_cast<int>(wParam);
        data->m_Keys[key] = true;
    } break;


    case WM_KEYUP:
    {
	    const int key = static_cast<int>(wParam);
        data->m_Keys[key] = false;
    } break;

    }
	
    return DefWindowProc(hWnd, msg, wParam, lParam);
}


#include "Utils.h"
constexpr uint32_t WIN_ALIGNMENT = 4;
uint32_t Window::GetAlignedWidth() const
{
    return alignUp(m_WindowData.m_Width, WIN_ALIGNMENT);
}

uint32_t Window::GetAlignedHeight() const
{
    return alignUp(m_WindowData.m_Height, WIN_ALIGNMENT);
}