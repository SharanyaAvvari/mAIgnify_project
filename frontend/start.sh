#!/bin/bash

# mAIstro Quick Start Script
# This script sets up and runs the complete mAIstro application

echo "=================================="
echo "🚀 mAIstro AI - Quick Start"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if Python is installed
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
print_success "Python $PYTHON_VERSION found"

# Check if directories exist
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    print_error "Backend or frontend directories not found!"
    print_info "Please ensure you have the correct project structure."
    exit 1
fi

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p uploads chat_histories trained_models
print_success "Directories created"

# Check for virtual environment
if [ ! -d "backend/venv" ]; then
    print_info "Creating Python virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Check for .env file
if [ ! -f ".env" ]; then
    print_warning ".env file not found!"
    print_info "Creating .env from template..."
    
    cat > .env << 'EOF'
# API Keys - Add your actual keys here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
COHERE_API_KEY=your_cohere_key_here
REPLICATE_API_KEY=your_replicate_key_here
FLASK_SECRET_KEY=your_secret_key_here
EOF
    
    print_warning "Please edit .env file and add your API keys!"
    print_info "Opening .env file..."
    ${EDITOR:-nano} .env
fi

# Install Python dependencies
print_info "Installing Python dependencies..."
cd backend
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
pip install -q -r requirements.txt
if [ $? -eq 0 ]; then
    print_success "Python dependencies installed"
else
    print_error "Failed to install Python dependencies"
    exit 1
fi
cd ..

# Check for Tesseract
print_info "Checking Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    TESSERACT_VERSION=$(tesseract --version 2>&1 | head -n 1)
    print_success "Tesseract found: $TESSERACT_VERSION"
else
    print_warning "Tesseract OCR not found!"
    print_info "Installing Tesseract..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y tesseract-ocr
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install tesseract
    else
        print_error "Please install Tesseract manually for your system"
    fi
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    print_info "Shutting down mAIstro..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    print_success "mAIstro stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend server
print_info "Starting backend server..."
cd backend
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    print_success "Backend server started (PID: $BACKEND_PID)"
else
    print_error "Failed to start backend server"
    exit 1
fi

# Start frontend server
print_info "Starting frontend server..."
cd frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 2

# Check if frontend is running
if ps -p $FRONTEND_PID > /dev/null; then
    print_success "Frontend server started (PID: $FRONTEND_PID)"
else
    print_error "Failed to start frontend server"
    kill $BACKEND_PID
    exit 1
fi

# Display information
echo ""
echo "=================================="
echo "✨ mAIstro AI is now running! ✨"
echo "=================================="
echo ""
print_info "Backend API:  http://localhost:8000"
print_info "Frontend:     http://localhost:3000/login.html"
echo ""
print_success "Login Page:   http://localhost:3000/login.html"
print_success "Main Chat:    http://localhost:3000/index.html"
echo ""
print_warning "Press Ctrl+C to stop all servers"
echo ""

# Keep script running
wait $BACKEND_PID $FRONTEND_PID