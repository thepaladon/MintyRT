#include "Window.h"

// Maybe ignore warnings in these include
#ifndef UNICODE
#define UNICODE
#endif 
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

    ShowWindow(m_WindowHandle, SW_SHOWDEFAULT);

    // Here I pass in struct with data to alter when needed
    SetWindowLongPtr(m_WindowHandle, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&m_WindowData));

    UpdateWindow(m_WindowHandle);

    m_Bitmap = CreateSampleDIB();


}

HBITMAP Window::CreateSampleDIB()
{
    // Create a 100x100 pixel DIB with RGB color
    BITMAPINFO bmi = { 0 };
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = m_WindowData.m_Width;
    bmi.bmiHeader.biHeight = -m_WindowData.m_Height;  // Negative height for top-down DIB
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 24;  // 24-bit RGB
    bmi.bmiHeader.biCompression = BI_RGB;

    void* pPixels = nullptr;
    HBITMAP hDIB = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, &pPixels, NULL, 0);

    // Populate the DIB with some sample data (e.g., a red square)
    if (hDIB) {
        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 100; x++) {
                BYTE* pPixel = reinterpret_cast<BYTE*>(pPixels) + ((y * 100 + x) * 3);
                pPixel[0] = 255; // Red
                pPixel[1] = 0;   // Green
                pPixel[2] = 0;   // Blue
            }
        }
    }

    m_BitmapInfo = bmi;
    m_fb = pPixels;
    m_Bitmap = hDIB;
    return hDIB;
}




bool Window::OnUpdate(void* fb)
{
    // Reset from last frame
    m_WindowData.m_Resized = false;

    //  Goes through all messages and events (to process input)
    MSG msg;

    while (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    if (m_Bitmap) {
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
        }
    }



    return m_WindowData.m_Alive;
}

void Window::Shutdown()
{
    m_WindowHandle = nullptr;
}


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

    } break;

    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}