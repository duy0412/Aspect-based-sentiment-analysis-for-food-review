import os
import json
import csv 
from openai import OpenAI


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key="keyhere") 

MODEL_NAME = "gpt-4o-latest-2025-01-29)" #

# --- Define Aspects and Sentiments based on the image ---

ASPECTS = ["FOOD", "SERVICE", "PRICE", "AMBIENCE"]
SENTIMENT_LABELS = ["positive", "negative", "none"] 
SENTIMENT_MAP_ID = {"positive": 1, "negative": 2, "none": 0}
SENTIMENT_MAP_LABEL = {v: k for k, v in SENTIMENT_MAP_ID.items()} # Map ID back to label if needed


# --- Define the Prompt Structure (Vietnamese) ---

def create_annotation_prompt_vietnamese(review_text):
    """Creates the prompt for GPT-4o with Vietnamese few-shot examples."""

    prompt = f"""
Phân tích đánh giá của khách hàng sau đây bằng tiếng Việt. Xác định cảm xúc ('positive', 'negative', 'none') cho từng khía cạnh được xác định trước: {', '.join(ASPECTS)}.

Chỉ xuất kết quả ở định dạng JSON với cấu trúc sau:
{{
  "ASPECT_NAME": "sentiment_label",
  ... (lặp lại cho tất cả các khía cạnh)
}}

--- VÍ DỤ MẪU (FEW-SHOT EXAMPLES) ---

**Ví dụ 1:**
Review: "ăn rất ngon được phục vụ chu đáo với 2 cô chú người huế có rất nhiều món như bò mọc chân giò chả cua rất đặc biệt các bạn nên thử thập cẩm là đầy đủ tất cả một bát chỉ có 30 nghìn đồng"
Desired JSON Output:
{{
  "FOOD": "positive",
  "SERVICE": "positive",
  "PRICE": "positive",
  "AMBIENCE": "none"
}}

**Ví dụ 2:**
Review: "quán này khá đông khách vào buổi tối nên phải đi sớm mới có chỗ được v món ăn ngon giá cả mềm phục vụ chắc do quán đông nên khá thờ ơ tý nhưng bỏ qua mấy điểm đó thì quán ngon lành d"
Desired JSON Output:
{{
  "FOOD": "positive",
  "AMBIENCE": "negative",
  "SERVICE": "negative",
  "PRICE": "positive"
}}

--- KẾT THÚC VÍ DỤ ---

**Bây giờ, phân tích đánh giá này:**
Review: "{review_text}"
Desired JSON Output:
"""
    return prompt

# --- Function to Call OpenAI API ---

def get_annotation_from_gpt(review):
    """Sends the review to GPT-4o for annotation and returns the parsed JSON."""

    prompt_content = create_annotation_prompt_vietnamese(review)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"Bạn là trợ lý chú thích chuyên nghiệp cho các đánh giá bằng tiếng Việt. Nhiệm vụ của bạn là Phân tích Cảm xúc dựa trên Khía cạnh (Aspect-Based Sentiment Analysis). Xác định cảm xúc ({', '.join(SENTIMENT_LABELS)}) cho các khía cạnh {', '.join(ASPECTS)}. Chỉ xuất ra định dạng JSON được chỉ định."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.2,
            max_tokens=200, # Reduced max_tokens as output is simpler
            response_format={"type": "json_object"}
        )

        response_content = response.choices[0].message.content

        try:
            parsed_json = json.loads(response_content)
            # Basic validation
            if not isinstance(parsed_json, dict):
                print(f"Warning: Response is not a JSON object: {response_content}")
                return None
            for aspect in ASPECTS:
                if aspect not in parsed_json:
                     print(f"Warning: Aspect '{aspect}' missing in response.")
                     # Handle missing aspect (e.g., assume 'none' or return error)
                elif parsed_json[aspect] not in SENTIMENT_LABELS:
                     print(f"Warning: Invalid sentiment '{parsed_json[aspect]}' for aspect '{aspect}'.")
                     # Handle invalid sentiment
            return parsed_json
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON response: {response_content}")
            return None

    except Exception as e:
        print(f"An error occurred calling the OpenAI API: {e}")
        return None

# --- Function to Format Output like the CSV ---

def format_to_csv_rows(original_review, annotation_json):
    """Formats the JSON annotation into a list of rows matching the CSV structure."""
    rows = []
    if not annotation_json:
        return rows

    for aspect in ASPECTS: # Iterate through predefined aspects to ensure order/completeness
        sentiment_label = annotation_json.get(aspect, "none") # Default to 'none' if aspect missing

        # Validate sentiment label
        if sentiment_label not in SENTIMENT_MAP_ID:
            print(f"Warning: Received unexpected sentiment label '{sentiment_label}' for aspect '{aspect}'. Defaulting to 'none'.")
            sentiment_label = "none"

        sentiment_id = SENTIMENT_MAP_ID[sentiment_label]

        rows.append([
            original_review,
            aspect,
            sentiment_id,
            sentiment_label
        ])
    return rows

# --- Example Usage ---

# Using one of the reviews from the examples for demonstration
vietnamese_review_1 = "ăn rất ngon được phục vụ chu đáo với 2 cô chú người huế có rất nhiều món như bò mọc chân giò chả cua rất đặc biệt các bạn nên thử thập cẩm là đầy đủ tất cả một bát chỉ có 30 nghìn đồng"
vietnamese_review_2 = "quán này khá đông khách vào buổi tối nên phải đi sớm mới có chỗ được v món ăn ngon giá cả mềm phục vụ chắc do quán đông nên khá thờ ơ tý nhưng bỏ qua mấy điểm đó thì quán ngon lành d"
# A new hypothetical review
new_vietnamese_review = "Đồ ăn tạm ổn, không đặc sắc lắm. Nhân viên khá chậm nhưng giá rẻ bất ngờ."


print(f"--- Annotating Review 1 ---")
print(f"Review: '{vietnamese_review_1}'")
annotation1 = get_annotation_from_gpt(vietnamese_review_1)
if annotation1:
    csv_rows1 = format_to_csv_rows(vietnamese_review_1, annotation1)
    print("Formatted Output Rows:")
    for row in csv_rows1:
        print(row)
else:
    print("Failed to get annotation.")


print(f"\n--- Annotating Review 3 (New) ---")
print(f"Review: '{new_vietnamese_review}'")
annotation3 = get_annotation_from_gpt(new_vietnamese_review)

if annotation3:
    csv_rows3 = format_to_csv_rows(new_vietnamese_review, annotation3)
    print("Formatted Output Rows:")
    for row in csv_rows3:
        print(row)

    output_filename = "annotations.csv"
    print(f"\nWriting annotations to {output_filename}...")
    try:
        # Use 'a' to append, 'w' to overwrite
        with open(output_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(['sentence', 'aspect', 'sentiment_id', 'sentiment'])
            writer.writerows(csv_rows3) # Write the rows for the current review
        print("Successfully wrote to CSV.")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

else:
    print("Failed to get annotation.")