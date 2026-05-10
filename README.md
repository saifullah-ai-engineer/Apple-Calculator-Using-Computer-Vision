# Apple Calculator Using Computer Vision - MathCam 📸

MathCam is a Flask + OpenCV web app that turns your webcam into an AI-assisted math canvas. Use hand gestures to draw an equation, save the canvas, and send the image to Gemini for a plain-text solution.

<p align="center">
  <img src="https://github.com/user-attachments/assets/03f7c640-820c-427c-98c8-d61728733853" height="375" alt="MathCam demo">
</p>

## Features

- Live webcam stream in a browser.
- MediaPipe hand tracking for selection and drawing gestures.
- Virtual pen colors and eraser toolbar.
- Save and clear controls from the UI.
- Gemini integration for solving handwritten math from the saved canvas.
- `.env`-based configuration so API keys are not hardcoded.

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/mediapipe-0097A7?style=for-the-badge&logo=google&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)

## Setup

1. Clone the repository and enter the project directory.

   ```bash
   git clone https://github.com/ayush-that/Apple-Calculator-Using-Computer-Vision.git
   cd Apple-Calculator-Using-Computer-Vision
   ```

2. Create and activate a Python virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Configure Gemini.

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and replace `your_api_key_here` with a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

5. Run the app.

   ```bash
   python app.py
   ```

6. Open <http://127.0.0.1:5000> and allow camera access in your browser.

## Gestures and Controls

- **Selection mode:** raise index + middle fingers and hover over the toolbar.
- **Drawing mode:** raise only the index finger.
- **Save gesture:** raise thumb + index finger.
- **Save button:** writes the current canvas to `saved_canvas.jpg`.
- **Clear button:** resets the virtual canvas.
- **Solve with Gemini:** sends the saved canvas to Gemini and displays the answer.
- **Quit desktop mode:** if using `python main.py`, press `q` in the OpenCV window.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | _required for solving_ | Gemini API key used by `/solve`. |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Gemini model used for image solving. |
| `FLASK_RUN_HOST` | `127.0.0.1` | Host used when running `python app.py`. |
| `PORT` | `5000` | Port used when running `python app.py`. |
| `FLASK_DEBUG` | `0` | Set to `1` to enable Flask debug mode. |

## Troubleshooting

- If the browser shows a camera error frame, confirm that your webcam is connected and no other application is using it.
- If solving returns an API key warning, verify that `.env` exists and contains `GOOGLE_API_KEY=...`.
- If MediaPipe or OpenCV fails to install, use a supported Python version for those wheels and recreate the virtual environment.

## License

Distributed under the GNU General Public License v3.0 License. See `LICENSE` for more information.
