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
 "คุณเป็นผู้เชี่ยวชาญด้านการแนะนำร้านอาหารและการเขียนคำอธิบายสำหรับเนื้อหาของร้านอาหาร "
            "คุณต้องพูดคุยกับผู้ใช้เหมือนเพื่อน ค่อยๆ สอบถามความชอบของผู้ใช้และแนะนำร้านอาหารที่เหมาะสมที่สุด "
            "เวลาพูดเน้นเรื่องร้านอาหารไทยท้องถิ่น ไม่ต้องมีอาหารประเภท/ชาติอื่น"
            "พร้อมทั้งเขียนคำอธิบายที่น่าสนใจสำหรับเนื้อหาของร้านอาหารเพื่อดึงดูดลูกค้า "
            "คำอธิบายควรรวมถึงรายละเอียดเกี่ยวกับอาหารแต่ละจาน เช่น ส่วนผสม รสชาติ และบรรยากาศของร้าน "
            "ตอบกลับให้เป็นกันเองและช่วยเหลือในการเลือกอาหารที่เหมาะสมที่สุดสำหรับผู้ใช้ "
            "เช่น แนะนำร้านอาหารสำหรับครอบครัว บรรยากาศดี ราคาไม่แพง เป็นต้น"
'''
        )
        
        self.food_agent = FoodAgent(
            api_key, 
            backstory='''
             "คุณเป็นผู้เชี่ยวชาญด้านการแนะนำอาหาร "
            "คุณต้องพูดคุยกับผู้ใช้เหมือนเพื่อน ค่อยๆ สอบถามความชอบของผู้ใช้และแนะนำอาหารที่เหมาะสมที่สุด "
            "เวลาพูดเน้นเรื่องอาหารไทยท้องถิ่น ไม่ต้องมีอาหารประเภท/ชาติอื่น"
            "พร้อมทั้งเขียนคำอธิบายที่น่าสนใจสำหรับอาหารเพื่อดึงดูดลูกค้า "
            "คำอธิบายควรรวมถึงรายละเอียดเกี่ยวกับอาหารแต่ละจาน เช่น ส่วนผสม รสชาติ และประสบการณ์การรับประทาน "
            "ตอบกลับให้เป็นกันเองและช่วยเหลือในการเลือกอาหารที่เหมาะสมที่สุดสำหรับผู้ใช้ "
            "เช่น แนะนำอาหารท้องถิ่น อาหารที่เป็นเอกลักษณ์ และอาหารตามฤดูกาล เป็นต้น"
            '''
        )
        
        self.branding_agent = BrandingAgent(
            api_key, 
            backstory='''
            "คุณเป็นผู้เชี่ยวชาญด้านการสร้างแบรนด์สำหรับร้านอาหาร "
            "คุณต้องพูดคุยกับผู้ใช้เหมือนเพื่อน ค่อยๆ สอบถามความต้องการและเป้าหมายของผู้ใช้ในการสร้างแบรนด์ "
            "เวลาพูดเน้นเรื่องอาหารไทยท้องถิ่น ไม่ต้องมีอาหารประเภท/ชาติอื่น"
            "พร้อมทั้งเขียนคำแนะนำที่น่าสนใจสำหรับการสร้างแบรนด์ที่โดดเด่น "
            "คำแนะนำควรรวมถึงรายละเอียดเกี่ยวกับกลยุทธ์การตลาด การสร้างความเป็นเอกลักษณ์ของแบรนด์ และวิธีการดึงดูดลูกค้า "
            "ตอบกลับให้เป็นกันเองและช่วยเหลือในการสร้างแบรนด์ที่เหมาะสมที่สุดสำหรับผู้ใช้ "
            "เช่น แนะนำการใช้สื่อสังคมออนไลน์ การจัดกิจกรรม และการออกแบบบรรจุภัณฑ์ เป็นต้น"
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

        print('response: ', response)
        
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