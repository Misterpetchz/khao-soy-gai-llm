import os
import json
from agents import PreprocessAgent, RetrievalAgent, RestaurantAgent, FoodAgent, BrandingAgent, PostprocessAgent
from typhoon import get_typhoon_response

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

api_key = os.getenv("TYPHOON_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENTYPHOON_API_KEY environment variable.")

class ManagerAgent:
    def __init__(self, api_key):
        self.preprocess_agent = PreprocessAgent()
        self.retrieval_agent = RetrievalAgent("data_special.csv")  # Pass the CSV file path to the RetrievalAgent
        
        self.restaurant_agent = RestaurantAgent(
            api_key, 
            backstory="You must answer only in Thai."
        )
        
        self.food_agent = FoodAgent(
            api_key, 
            backstory="You must answer only in Thai."
        )
        
        self.branding_agent = BrandingAgent(
            api_key, 
            backstory="You must answer only in Thai."
        )
        
        self.postprocess_agent = PostprocessAgent()

    def predict_agent(self, data):
        # # Simple heuristic-based method to predict the agent based on keywords
        # data_lower = data.lower()
        # if "อาหาร" in data_lower or "ข้าวซอย" in data_lower or "แกงฮังเล" in data_lower:
        #     return 'food'
        # elif "ร้านอาหาร" in data_lower or "บรรยากาศ" in data_lower:
        #     return 'restaurant'
        # elif "แบรนด์" in data_lower or "การตลาด" in data_lower:
        #     return 'branding'
        # else:
        #     return 'unknown'
        return get_typhoon_response(user_text = data.lower(), api_key = api_key, backstory = "You are a helpful assistant. Classify the following Thai text into one of the three classes - food, restaurant, or branding. Only respond with the category name.")
        

    def handle_request(self, request):
        try:
            data = request.get("data")

            if not data:
                return {"error": "Missing data"}

            # Predict the appropriate agent based on the data content
            task_type = self.predict_agent(data)

            if task_type == 'unknown':
                return {"error": "Unable to determine the appropriate agent"}

            # Preprocess the data
            preprocessed_data = self.preprocess_agent.preprocess(data)

            # Perform retrieval to get relevant information
            retrieved_info = self.retrieval_agent.retrieve_information(preprocessed_data)

            if task_type == 'restaurant':
                response = self.restaurant_agent.handle_task(preprocessed_data, retrieved_info)
            elif task_type == 'food':
                response = self.food_agent.handle_task(preprocessed_data, retrieved_info)
            elif task_type == 'branding':
                response = self.branding_agent.handle_task(preprocessed_data, retrieved_info)
            else:
                return {"error": "Invalid task type"}

            # Postprocess the response
            final_response = self.postprocess_agent.postprocess(response)
            return final_response

        except Exception as e:
            return {"error": str(e)}
       
# Example Usage
if __name__ == "__main__":
    manager = ManagerAgent(api_key)

    # Create a sample request for a local Thai food description (Chiang Mai)
    request = {
        "data": "ช่วยแนะนำร้านอาหารท้องถิ่นในเชียงใหม่ที่มีชื่อเสียงและเป็นที่นิยม โดยเน้นร้านที่เสิร์ฟอาหารพื้นเมืองของเชียงใหม่ เช่น ข้าวซอย แกงฮังเล และน้ำพริกอ่อง และบรรยากาศดีสำหรับครอบครัว กรุณาตอบเป็นภาษาไทย"
    }

    # Handle the request
    response = manager.handle_request(request)
    print(json.dumps(response, indent=4, ensure_ascii=False))