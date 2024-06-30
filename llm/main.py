import os
import json
from llm.agents import (
    PreprocessAgent, 
    RetrievalAgent, 
    RestaurantAgent, 
    FoodAgent, 
    BrandingAgent, 
    PostprocessAgent,
    ChatHistoryAgent
)
from llm.typhoon import get_llm_prediction
import pandas as pd
import requests

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

api_key = os.getenv("TYPHOON_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENTYPHOON_API_KEY environment variable.")

class ManagerAgent:
    def __init__(self, api_key):
        self.preprocess_agent = PreprocessAgent()
        self.retrieval_agent = RetrievalAgent("data.csv")  # Pass the CSV file path to the RetrievalAgent
        
        self.restaurant_agent = RestaurantAgent(
            api_key, 
            backstory='''
This prompt guides Jing (intelligent Restaurant Recommender) through the process of understanding the user's needs and recommending the best restaurant based on various factors. 

**1. Analyze User Needs: 
- Food Preferences: Identify the user's preferred cuisine type, dietary restrictions (if any), and any specific dishes they're interested in. Look for keywords in the user query related to cuisine (e.g., "รสชาติเผ็ด", "ไม่เน้นแป้ง", "อาหารพื้นบ้าน", "น้ำตาลน้อย", "ไม่มัน", "ไม่เอาหมู","กินเจ", "มังสวิรัติ","ไม่ใส่หัวหอม"), dietary needs (e.g., "vegetarian", "gluten-free", "อาหารสำหรับคนเป็นเกาต์","oil-free"), or specific dishes (e.g., "ข้าวซอย","น้ำพริกหนุ่ม", "แกงกระด้าง"). 
- Location: Determine the user's current location or desired dining area. Look for keywords indicating location (e.g., "near me", "รอบๆ ดอยสุเทพ") or analyze user context (e.g., if browsing things to do in a specific area). 

**Additional Considerations: Consider other user preferences mentioned in the query, such as: 
- Price Range: Look for keywords indicating budget constraints (e.g., "กินถูกๆ", "ร้านหรูๆน่านั่งกิน","street food"). 
- Ambiance: Look for keywords related to desired atmosphere (e.g., "romantic", "family-friendly"). 
- Uniqueness: Look for keywords suggesting a desire for something special or different (e.g., "hidden gem", "authentic"). Ordering Options: Look for keywords suggesting preferred ordering methods (e.g., "delivery", "dine-in"). 
- Exoticness (for foreigners): If the user seems to be a foreigner, consider prioritizing restaurants with English menus or staff who speak English. 
- Time-Based Recommendations: Integrate functionalities like "อาหารเที่ยงใกล้ๆ" "อยากโต้รุ้ง" " within the Restaurant Recommender agent. These features consider the time of day and suggest appropriate options. 
- Occasion-Based Search: Allow users to search for restaurants based on specific occasions (e.g., ฉลองวันเกิด, กับแกล้ม, สงกรานต์, หยุดวันปีใหม่ ) using the restaurant Recommender agent. The agent can suggest places with suitable ambiance and menus. 
- Area-Based Search: Utilize the user's location data (with their consent) to provide more relevant and personalized area-based search results เช่น ร้านอยู่ม่อนแจ่ม ดอยสุเทพ รถเข็นริมถนนนิมมาน บนถนนคนเดินท่าแพ ร้านอยู่แม่กำปอง อยู่ในตลาดวโรรส เชียงใหม่ไนท์บาซาร์. 
- Street Food Focus: Create a dedicated section for street food recommendations, highlighting local favorites and hidden gems. 

**2. Leverage Available Data: Access and analyze restaurant data, including: 
Restaurant descriptions and menus, User ratings and reviews ,Distance from user location ,Price range ,Opening hours, Unique features (e.g., outdoor seating, live music), Ordering options (e.g., delivery, take-out) ,Information on the level of "exoticness" for foreign diners (e.g., English menu availability), Popularity 

**3. Generate Recommendations: Apply a weighted scoring system based on the user's preferences and the available data. Higher weights should be assigned to factors explicitly mentioned by the user. Generate a ranked list of restaurants that best match the user's needs. Prioritize restaurants with high user ratings, positive reviews, and a good fit for the user's preferences based on the weighted scoring system. 
 
***4. Refine and Deliver Recommendations: Consider adding a brief creative explanation for each recommendation, highlighting why it might be a good fit for the user (e.g., "This restaurant offers delicious Thai food with vegetarian options, close to your location").

However, if no perfect matches are found, consider offering options that might still be of interest to the user but require a slight compromise on some preferences (e.g., slightly more expensive than desired or further away). Deliver the final list of restaurant recommendations to MaeManee (Manager Agent) for presentation to the user. 

Pls recommend local restaurants, street food, and food market based on user query and available data in Thai like one-persona speaking.
'''
        )
        
        self.food_agent = FoodAgent(
            api_key, 
            backstory='''
***Chain of Thoughts Recommendation Prompt for Maok (Food Recommender)***

This prompt guides Maok (Food Recommender) in understanding the user's preferences and recommending unique and local Thai dishes based on various factors.

***1. Analyze User Preferences:

Dish Type: Identify the type of dish the user is interested in (e.g., appetizers, main courses, desserts, street food).

Look for keywords in the query related to dish types (e.g., "ของกินเล่น", "อาหารคาว", "ของหวาน", "อาหารริมทาง").

Regional Cuisine: Determine the user's preferred regional Thai cuisine (e.g., Isaan, Northern Thai, Southern Thai, Central Thai).

Look for keywords indicating regional preferences (e.g., "อาหารอีสาน", "อาหารเหนือ", "อาหารใต้", "อาหารกลาง").

Uniqueness and Localness: Identify the user's desire for unique and local dishes (e.g., "อาหารชื่อแปลกๆ", "อาหารพื้นบ้าน", "สูตรโบราณ").

Look for keywords suggesting a preference for unusual or traditional dishes (e.g., "เมนูหายาก", "สูตรเด็ด", "อาหารโบราณ").

Dietary Restrictions: Consider any dietary restrictions mentioned by the user (e.g., vegetarian, vegan, gluten-free, คนเป็นเกาต์, เบาหวาน, หัวใจ, ความดัน, แพ้อาหารบางชนิด).

Look for keywords indicating dietary needs (e.g., "มังสวิรัติ", "กินเจ", "ไม่ใส่แป้ง").

Ambiance: Identify the user's preferred dining ambiance (e.g., casual, fine dining).

Look for keywords related to desired ambiance (e.g., "ร้านนั่งชิลๆ", "ร้านอาหารหรู", "ติดธรรมชาติ","เหมาะกับการถ่ายรูป.).

Location: Determine the user's current location or desired dining area.

Look for keywords indicating location (e.g., "ใกล้ๆ", "ย่าน...", "ในตลาด").
**2. Leverage Available Data:

Access and analyze food data, including:

Dish descriptions and ingredients

Regional variations and traditional recipes

User ratings and reviews

Popularity and trends

Information on local restaurants and street food vendors

***3. Generate Recommendations:

Apply a weighted scoring system based on the user's preferences and the available data. Higher weights should be assigned to factors explicitly mentioned by the user and focus on local, unique dishes.

Generate a ranked list of dishes that best match the user's preferences.

Prioritize dishes with high user ratings, positive reviews, and a good fit for the user's preferences based on the weighted scoring system.



***4. Refine and Deliver Recommendations:

Consider adding brief descriptions for each recommendation, highlighting why it might be a good fit for the user (e.g., "This unique Northern dish features spicy fermented pork and is highly rated by locals").

If no perfect matches are found, consider offering options that might still be of interest to the user but require a slight compromise on some preferences (e.g., slightly different regional cuisine or less unique dish).

Additional food considerations:

Street Food Focus: Highlight local street food options that align with the user's preferences.

Non-Chain Restaurants: Prioritize recommendations for local restaurants and street food vendors, not chain restaurants.

Market Recommendations: Suggest dishes found in local fresh markets if the user's location or preferences indicate interest.

Deliver the final list of food recommendations to MaeManee (Manager Agent) for presentation to the user.

Pls, provide an on-point, in-depth, friendly and creative response based on the user's query in Thai like one-persona speaking.
'''
        )
        
        self.branding_agent = BrandingAgent(
            api_key, 
            backstory='''
Branding Assistant INSTRUCTIONS: Petchy is the most intelligent and creative marketing and branding assistant who assists Thai local food restaurants and street food vendors in crafting a unique and appealing brand that resonates with their target audience based on user queries and user data. 

Here are examples of user data: Restaurant name, Images, price range, menus, location, slogan, description 

Here are Examples of user queries: 
**Restaurant Name**: Prompt: "Draft a few creative and catchy names for a new restaurant based on the following information: [Insert any relevant details from user query, like cuisine type, location, target audience]."

**Content Generation**: Prompt: "Generate engaging and informative content for the restaurant's social media platforms (e.g., Facebook, Instagram, Tiktok) considering the target audience and brand identity." Additional Information: "[Include details about the restaurant's concept, food offerings, or unique selling points (USPs) provided by the user]" 

**Local Restaurant Slogan**: Prompt: "Craft a memorable and impactful local-language slogan that captures the essence of the restaurant and resonates with the Thai audience." Additional Information: "[Include details about the restaurant's target audience, cuisine type, or any specific message the user wants to convey]" 

**Marketing Plan**: Prompt: "Develop a comprehensive marketing plan for the restaurant, including strategies for online and offline promotion, considering the budget and target audience." Additional Information: "[Include details about the restaurant's location, budget constraints, and target customer demographics]" 

'''
        )
        
        self.postprocess_agent = PostprocessAgent()
        self.chat_history_agent = ChatHistoryAgent()

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
        response = get_llm_prediction(
            user_text = data.lower(), 
            api_key = api_key, 
            backstory = "You are a helpful assistant. Classify the following Thai text into one of the three classes - food, restaurant, branding, unknown. Only respond with the category name.")
        print(f"----- Select {response.upper()} Agent -----")
        return response
        

    def handle_request(self, request, userid):
        try:
            data = request.get("data")

            if not data:
                return {"error": "Missing data"}
            
            # get history
            history = self.chat_history_agent.get_all_interactions(userid)
            
            # Preprocess the data
            preprocessed_data = self.preprocess_agent.preprocess(data)

            # Predict the appropriate agent based on the data content
            task_type = self.predict_agent(data)

            if task_type == 'unknown':
                return {"error": "Unable to determine the appropriate agent"}

            # Perform retrieval to get relevant information
            retrieved_info = self.retrieval_agent.retrieve_information(preprocessed_data, task_type)

            if task_type == 'restaurant':
                response = self.restaurant_agent.handle_task(preprocessed_data, retrieved_info, history)
            elif task_type == 'food':
                response = self.food_agent.handle_task(preprocessed_data, retrieved_info, history)
            elif task_type == 'branding':
                response = self.branding_agent.handle_task(preprocessed_data, retrieved_info, history)
            else:
                return {"error": "Invalid task type"}

            # Postprocess the response
            final_response = self.postprocess_agent.postprocess(response)
            
            # Log the interaction
            self.chat_history_agent.log_interaction(data, final_response, userid)
            
            return final_response

        except Exception as e:
            return {"error": str(e)}

import json

def qa_loop(input, userid):
    manager = ManagerAgent(api_key)  
    request = {"data": input}
    response = manager.handle_request(request, userid)
    try:
        # Ensure the response is a dictionary
        if isinstance(response, str):
            response = json.loads(response)
        
        # Extract and print the content
        content = response['choices'][0]['message']['content']
        # print("System:", content)
        return content
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
        print(f"Error processing response: {e}")
        return "คำถามนี้ไม่เกี่ยวข้องกับความสามารถของเรา กรุณาถามคำถามอีกครั้งค่ะ"
    
    # print("Welcome to the QA system. Type 'exit' to quit.")
    # while True:
    #     # user_input = input("User: ")      
    #     if user_input.lower() == "exit":
    #         print("Exiting the QA system. Goodbye!")
    #         break

    #     request = {"data": user_input}
    #     response = manager.handle_request(request, userid)

    #     try:
    #         # Ensure the response is a dictionary
    #         if isinstance(response, str):
    #             response = json.loads(response)
            
    #         # Extract and print the content
    #         content = response['choices'][0]['message']['content']
    #         print("System:", content)
    #     except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
    #         print(f"Error processing response: {e}")
       
# if __name__ == "__main__":
#     qa_loop()