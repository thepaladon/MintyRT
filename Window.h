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

	float m_MouseDeltaX;
	float m_MouseDeltaY;

	int m_MouseGlobalPosX;
	int m_MouseGlobalPosY;

	bool m_Keys[0xff];
};

class Window
{
public:

	Window(uint32_t width, uint32_t height, std::string name);
	~Window() {};

	// Is false when window has been destroyed
	bool OnUpdate();
	void RenderFb(void* fb);
	void Shutdown();

	HBITMAP CreateSampleDIB();

	void SetWidth(uint32_t width) { m_WindowData.m_Width = width; }
	void SetHeight(uint32_t height) { m_WindowData.m_Height = height; }

	uint32_t GetHeight() const { return m_WindowData.m_Height; }
	uint32_t GetWidth() const { return m_WindowData.m_Width; }
	float GetMouseDeltaX() const { return m_WindowData.m_MouseDeltaX; }
	float GetMouseDeltaY() const { return m_WindowData.m_MouseDeltaY; }
	bool GetKey(int key) const { return m_WindowData.m_Keys[key]; }
	uint32_t GetAlignedWidth() const;
	uint32_t GetAlignedHeight() const;
	bool GetIsResized() const { return m_WindowData.m_Resized; }

private:

	HBITMAP m_Bitmap = nullptr;
	BITMAPINFO m_BitmapInfo = {};

	WindowData m_WindowData = WindowData();
	HWND m_WindowHandle;

};