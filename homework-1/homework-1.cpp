// hw3-windows.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "homework-1.h"
#include "FreeImage.h"
#include "Scene.h"
#include "RayTracer.h"
#include "OptiXTracer.h"

#define MAX_LOADSTRING 100



// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name
LPCTSTR DefaultInputFile = L"test.txt";

// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
BOOL				InitStatusBar();
void				SetStatusText(LPCTSTR text);
void				SetClientSize(int width, int height);
void				DrawToScreen(HDC hdc);
int					LoadScene(LPCTSTR file);
void				ShowError(int err);
int					SaveImage(const char* filename, int width, int height, COLORREF* colorBuffer);
DWORD WINAPI		TracerThread(LPVOID lpParam);
DWORD WINAPI		ProgressThread(LPVOID lpParam);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);


RayTracer tracer;
OptiXTracer optixTracer;

Scene scene;
HWND hwndMain, hwndStatusBar; 
HANDLE tracerThread;

int main() {
	return _tWinMain(GetModuleHandle(NULL), NULL, GetCommandLine(), SW_SHOW);
}

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

	FreeImage_Initialise();


    // Initialize global strings
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_HW3WINDOWS, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_HW3WINDOWS));

	InitStatusBar();
	SetStatusText(L"Loading...");

	tracerThread = 0;

	LPCTSTR fileName = DefaultInputFile;
	//LPCTSTR fileName = lpCmdLine;
	//if(_tcslen(fileName) == 0)
	//	fileName = DefaultInputFile;

	int err = LoadScene(fileName);
	if (err == 0) {
		SetClientSize(scene.width, scene.height);

		DWORD threadId;
		tracerThread = CreateThread(
			NULL,                   // default security attributes
			0,                      // use default stack size  
			TracerThread,	        // thread function name
			NULL,		            // argument to thread function 
			0,                      // use default creation flags 
			&threadId);		        // returns the thread identifier

	}
	else {
		ShowError(err);
	}

    MSG msg;

    // Main message loop:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

	if(tracerThread != 0)
		WaitForSingleObject(tracerThread, 3000);

	//tracer.Cleanup();
	optixTracer.Cleanup();

	FreeImage_DeInitialise();

    return (int) msg.wParam;
}



//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_HW3WINDOWS));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_HW3WINDOWS);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	hInst = hInstance; // Store instance handle in our global variable

	DWORD windowStyle = WS_OVERLAPPEDWINDOW & (~WS_MAXIMIZEBOX) & (~WS_THICKFRAME);

	HWND hWnd = CreateWindowW(szWindowClass, szTitle, windowStyle,
		CW_USEDEFAULT, 0, RT_WIDTH, RT_HEIGHT, nullptr, nullptr, hInstance, nullptr);

	if (!hWnd)
	{
		return FALSE;
	}

	hwndMain = hWnd;

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	return TRUE;
}

BOOL InitStatusBar() {
	InitCommonControls();

	int idStatus = 2;

	// Create the status bar.
	hwndStatusBar = CreateWindowEx(
		0,                       // no extended styles
		STATUSCLASSNAME,         // name of status bar class
		(PCTSTR)NULL,            // no text when first created
		WS_CHILD | WS_VISIBLE,   // creates a visible child window
		0, 0, 0, 0,              // ignores size and position
		hwndMain,                // handle to parent window
		(HMENU)idStatus,         // child window identifier
		hInst,                   // handle to applicion ation instance
		NULL);                   // no window creatdata
	
	SendMessage(hwndStatusBar, SB_SIMPLE, TRUE, 0);

	return TRUE;
}

void SetStatusText(LPCTSTR text) {
	SendMessage(hwndStatusBar, SB_SETTEXT, SB_SIMPLEID, (LPARAM)text);
	OutputDebugString(text);
	OutputDebugString(L"\n");
}

void SetClientSize(int width, int height) {
	RECT rcStatus, mainStatus, offset;

	// Get status bar size
	GetWindowRect(hwndStatusBar, &rcStatus);
	GetWindowRect(hwndMain, &mainStatus);

	offset.left = rcStatus.left - mainStatus.left;
	offset.top = rcStatus.top - mainStatus.top;
	offset.right = rcStatus.right - mainStatus.right;
	offset.bottom = rcStatus.bottom - mainStatus.bottom;


	DWORD windowStyle = WS_OVERLAPPEDWINDOW & (~WS_MAXIMIZEBOX) & (~WS_THICKFRAME);

	RECT rect;
	rect.bottom = mainStatus.top + height + (rcStatus.bottom - rcStatus.top);
	rect.left = mainStatus.left;
	rect.top = mainStatus.top;
	rect.right = mainStatus.left + width;
	AdjustWindowRect(&rect, windowStyle, true);


	MoveWindow(hwndMain, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, TRUE);
	MoveWindow(hwndStatusBar, rect.left + offset.left, rect.top + offset.top, 
		rect.right - rect.left - offset.left + offset.right, 
		rect.bottom - rect.top - offset.top + offset.bottom, TRUE);
}

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_COMMAND:
        {
            int wmId = LOWORD(wParam);
            // Parse the menu selections:
            switch (wmId)
            {
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break;
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            
			DrawToScreen(hdc);
			OutputDebugString(L"WM_PAINT\n");

            EndPaint(hWnd, &ps);
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

int LoadScene(LPCTSTR file) {
	return scene.Parse(file);
}

void ShowError(int err) {
	switch (err) {
	case ERR_INVALID_FILE:
		SetStatusText(L"Invalid input file.");
		break;

	case ERR_FILE_NOT_FOUND:
		SetStatusText(L"File not found.");
		break;

	default:
		SetStatusText(L"Unknown error.");
	}
}

void DrawToScreen(HDC hdc) {
	COLORREF *arr = (COLORREF*)calloc(scene.width * scene.height, sizeof(COLORREF));
	
	/* Filling array here */
	//tracer.Fill(arr);
	optixTracer.Fill(arr);


	// Creating temp bitmap
	HBITMAP map = CreateBitmap(scene.width, scene.height, 1, 8 * 4,  (void*)arr);

	
	HDC src = CreateCompatibleDC(hdc); // hdc - Device context for window, I've got earlier with GetDC(hWnd) or GetDC(NULL);
	
	SelectObject(src, map); // Inserting picture into our temp HDC

	// Copy image from temp HDC to window
	BitBlt(hdc, // Destination
		0,  // x and
		0,  // y - upper-left corner of place, where we'd like to copy
		scene.width, // width of the region
		scene.height, // height
		src, // source
		0,   // x and
		0,   // y of upper left corner  of part of the source, from where we'd like to copy
		SRCCOPY); // Defined DWORD to juct copy pixels. Watch more on msdn;

	DeleteObject(map);
	DeleteDC(src); // Deleting temp HDC


	if (scene.outputFileName.length() > 0)
		SaveImage(scene.outputFileName.c_str(), scene.width, scene.height, arr);


	free(arr);
}


DWORD WINAPI TracerThread(LPVOID lpParam) {

	DWORD progresThreadId;
	HANDLE progressThread = CreateThread( NULL, 0, ProgressThread, NULL, 0, &progresThreadId);

	//tracer.Trace(scene);
	optixTracer.Trace(scene);

	RedrawWindow(hwndMain, NULL, NULL, RDW_INVALIDATE);

	WaitForSingleObject(progressThread, 1000);

	SetStatusText(L"Ready");

	return 0;
}


int SaveImage(const char* filename, int width, int height, COLORREF* colorBuffer) {

	BYTE * bytes = (BYTE*)malloc(width * height * 3);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int offset = (y * width + x) * 3;
			int cbOffset = (y * width + x);
			bytes[offset + 2] = (BYTE)((colorBuffer[cbOffset] & 0x00FF0000) >> 16);
			bytes[offset + 1] = (BYTE)((colorBuffer[cbOffset] & 0x0000FF00) >> 8);
			bytes[offset] = (BYTE)((colorBuffer[cbOffset] & 0x000000FF));
		}
	}

	FIBITMAP *img = FreeImage_ConvertFromRawBits((BYTE*)bytes, width, height, width * 3,
		24, 0xFF0000, 0x00FF00, 0x0000FF, true);

	free(bytes);

	return FreeImage_Save(FIF_PNG, img, filename, 0);
}



DWORD WINAPI ProgressThread(LPVOID lpParam) {
	/*
	while(tracer.progress < 1) {
		int progress = (int)(tracer.progress * 100.0f);
		WCHAR buffer[100];
		swprintf_s(buffer, L"Loading  |  Progress: %d%%", progress);
		SetStatusText(buffer);
		Sleep(500);
	}
	*/

	return 0;
}


// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
