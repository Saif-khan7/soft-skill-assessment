# soft-skill-assessment

🛠️ Setup Instructions
1️⃣ Clone the Repository

2️⃣ Set Up Backend
Step 1: Create a virtual environment
cd backend
python -m venv venv

Step 2: Activate the environment
Windows (PowerShell):
.\venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

Step 3: Install required Python packages
pip install flask flask-cors opencv-python deepface whisper

Step 4: Run the Flask app
python app.py
✅ The backend should now be running at: http://localhost:5000

3️⃣ Set Up Frontend (React)
Step 1: Open a new terminal and navigate to frontend
cd deepface-react-app

Step 2: Install dependencies
npm install

Step 3: Start the React app
npm start
✅ The frontend should now be running at: http://localhost:3000
