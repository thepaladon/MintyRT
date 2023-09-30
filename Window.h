#pragma once

#include <string>
#include <Windows.h>

// In a struct since I need to pass this data to wndproc function as a pointer
struct WindowData
{
	// Used to trigger things on window events
	// cpp is responsible to set these values when needed
	bool m_Alive = true;
	bool m_Resized = false;

	uint32_t m_Width;
	uint32_t m_Height;
};

class Window
{
public:

	Window(uint32_t width, uint32_t height, std::string name);
	~Window() {};

	// Is false when window has been destroyed
	bool OnUpdate(void* fb);
	void Shutdown();

	HBITMAP CreateSampleDIB();
	uint32_t GetHeight() { return m_WindowData.m_Height; }
	uint32_t GetWidth() { return m_WindowData.m_Width; }

	void SetWidth(uint32_t width) { m_WindowData.m_Width = width; }
	void SetHeight(uint32_t height) { m_WindowData.m_Height = height; }

	bool GetHasResized() { return m_WindowData.m_Resized; }


private:

	HBITMAP m_Bitmap = nullptr;
	BITMAPINFO m_BitmapInfo = {};
	void* m_fb = nullptr;

	WindowData m_WindowData = WindowData();
	HWND m_WindowHandle;

};