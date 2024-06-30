import os
from dotenv import load_dotenv
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    StickerMessage,
    QuickReply,
    QuickReplyItem,
    FlexMessage,
    FlexBubble,
    FlexBox,
    FlexButton,
    FlexText,
    FlexSeparator,
    MessageAction,
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    StickerMessageContent
)
from utils.process import query_place_collection
from typhoon_integration import generate_with_typhoon
from llm.main import qa_loop

# Load environment variables
load_dotenv()
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')

# Initialize WebhookHandler
handler = WebhookHandler(LINE_CHANNEL_SECRET)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    text = event.message.text
    userid = event.source.user_id
    generated_response = qa_loop(text, userid)
    # recommendations = query_place_collection(text, 2)
    # generated_response = generate_with_typhoon(text, recommendations, userid)
        
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
            
        if text == "แม่มณี":
            # Create Flex buttons
            flex_buttons = [        
                FlexSeparator(margin="lg"),           
                FlexButton(
                    style='primary',
                    color='#F7B62B',
                    action=MessageAction(label="ร้านอาหารใกล้ฉัน", text="ร้านอาหารใกล้ฉัน")
                ),
                FlexSeparator(margin="lg"),
                FlexButton(
                    style='primary',
                    color='#F7B62B',
                    action=MessageAction(label="เมนูยอดนิยม", text="เมนูยอดนิยมในไทย")
                ), 
                FlexSeparator(margin="lg"),
                FlexButton(
                    style='primary',
                    color='#F7B62B',
                    action=MessageAction(label="ค้นหาร้านอาหารตามจังหวัด", text="ค้นหาร้านอาหารตามจังหวัด")
                ),
                FlexSeparator(margin="lg"),
                FlexButton(
                    style='link',
                    color='#F7B62B',
                    action=MessageAction(label="อื่นๆ", text="อื่นๆ")
                ),                 
            ]

            # Create a Bubble container
            bubble = FlexBubble(
                body=FlexBox(
                    layout='vertical',
                    contents=[
                        FlexText(text="หัวข้อแนะนำ", weight="bold", size="lg"),                       
                        *flex_buttons
                    ]
                )
            )
            
            # Create FlexMessage
            flex_message = FlexMessage(alt_text="Flex Buttons", contents=bubble)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[flex_message]
                )
            )
        elif text == "บริการนี้ยังไม่เปิดให้ใช้งาน":
            return
        else:
            # QuickReply
            quick_reply_buttons = QuickReply(
                items=[
                    QuickReplyItem(
                        action=MessageAction(label="เรียกเจ๊จอง", text="เจ๊จอง")
                    ),
                    QuickReplyItem(
                        action=MessageAction(label="ร้านอาหารยอดฮิต", text="ร้านอาหารยอดฮิตในไทย")
                    ),
                ]
            )
            
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=generated_response, quick_reply=quick_reply_buttons)]
                )
            )

@handler.add(MessageEvent, message=StickerMessageContent)
def handle_sticker_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    StickerMessage(
                        package_id=event.message.package_id,
                        sticker_id=event.message.sticker_id
                    )
                ]
            )
        )