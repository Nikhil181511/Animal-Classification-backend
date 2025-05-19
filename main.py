from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torch.nn.functional import softmax
from PIL import Image
import torch
from torchvision import models, transforms
import io
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

GEMINI_KEY = "AIzaSyASuqtC0x-oMcRmmIDwY9h5_K-eDXnAoVI"
genai.configure(api_key=GEMINI_KEY)

# Gemini 1.5 flash model
modelchat = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("animal_classifier.pth", map_location=device)

class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            class_name = class_names[predicted.item()]
            confidence_score = round(confidence.item() * 100, 2)

        return JSONResponse({
            "prediction": class_name,
            "confidence": f"{confidence_score}%"  # e.g., "92.57%"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def format_response_to_bullets(response_text):
    lines = response_text.split("\n")
    bullet_points = "\n".join([f"• {line.strip()}" for line in lines if line.strip()])
    return bullet_points

def get_info_from_gemini(class_name):
    try:
        response = modelchat.generate_content(f"Tell me about the animal {class_name}")
        if response and hasattr(response, 'text'):
            formatted_response = format_response_to_bullets(response.text)
            return formatted_response
        else:
            return "No relevant information found."
    except Exception as e:
        return f"Error retrieving information: {str(e)}"

# POST endpoint for chatting and fetching animal details
@app.post("/chat")
async def chat(data: dict):
    user_message = data.get("text", "")
    predicted_animal = data.get("prediction", "") 

    context = (
        f"You are an expert animal assistant AI. The user is inquiring about animals. "
        f"The main animal in focus is: **{predicted_animal}**.\n"
        f"Provide detailed, helpful, and well-structured responses, especially about {predicted_animal}. "
        f"You also have knowledge about all other animals and should include comparative or related insights if relevant.\n\n"
        f"User's message: {user_message}\n\n"
        f"Reply in bullet points where appropriate, and keep the tone helpful and informative."
    )

    try:
        response = modelchat.generate_content(context)
        response_lines = response.text.split("\n")
        limited_response = "\n".join(response_lines[:8]) 
        formatted_response = format_response_to_bullets(limited_response)
        return {"response": formatted_response}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}

