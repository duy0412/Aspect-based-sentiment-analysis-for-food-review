import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API"))

model = genai.GenerativeModel('gemini-2.0-flash')

def get_train_prompt():
    prompt = """Now I am having this task: from a review of a customer in Vietnamese, identify the sentiment of 4 aspects in this order: food, ambience, service, price. Each has 3 rating: positive, neutral, negative. These are example.
    Example 1: 
    Input 1: ăn rất ngon được phục vụ chu đáo với 2 cô chú người huế có rất nhiều món như bò mọc chân giò chả cua rất đặc biết các bạn nên thử thập cẩm là đầy đủ tất cả một bát chỉ có 30 nghìn đồng 
    Ouput 1: 
    #1
    ăn rất ngon được phục vụ chu đáo với 2 cô chú người huế có rất nhiều món như bò mọc chân giò chả cua rất đặc biết các bạn nên thử thập cẩm là đầy đủ tất cả một bát chỉ có 30 nghìn đồng 
    {FOOD,positive}, {SERVICE, positive}, {PRICE, positive}

    Example 2: 
    Input 2: quán này khá đông khách vào buổi tối nên phải đi sớm mới có chỗ được v món ăn ngon giá cả mềm phục vụ chắc do quán đông nên khá thờ ơ tý nhưng bỏ qua mấy điểm đó thì quán ngon lành d
    Output 2:
    #2
    quán này khá đông khách vào buổi tối nên phải đi sớm mới có chỗ được v món ăn ngon giá cả mềm phục vụ chắc do quán đông nên khá thờ ơ tý nhưng bỏ qua mấy điểm đó thì quán ngon lành d
    {FOOD, positive}, {AMBIENCE, negative}, {SERVICE: negative}, {PRICE, positive}

    Example 3: 
    Input 3: nếu như muốn ăn các món đặc sản của miền bắc việt nam thì đây là địa điểm mà các bạn nên lưu lại menu phong phú giá cả bình dân muốn chắc chắn có bàn để thưởng thức thì các bạn sử dụng dịch vụ tablenow để book bàn trước nhé ngoài ra có thể đặt giao hàng tận nhà quá tiện lợi mà giá cả không thay đổi
    Output 3:
    #3
    nếu như muốn ăn các món đặc sản của miền bắc việt nam thì đây là địa điểm mà các bạn nên lưu lại menu phong phú giá cả bình dân muốn chắc chắn có bàn để thưởng thức thì các bạn sử dụng dịch vụ tablenow để book bàn trước nhé ngoài ra có thể đặt giao hàng tận nhà quá tiện lợi mà giá cả không thay đổi
    {PRICE, positive}, {SERVICE, positive}

    Example 4: 
    Input 4: nha hang phuc vu cac mon an rat ngon nha
    Output 4:
    #4
    nha hang phuc vu cac mon an rat ngon nha
    {FOOD, positive}

    Example 5:
    Input 6: nhiều ngày qua tình hình thế giới vô cùng căng thẳng vì những xung đột quân sự giữa mỹ và iran nhiều người lo lắng tình hình sẽ xấu đi và dẫn đến chiến tranh thế giới thứ 3
    Output 6: (blank)

    Example 6:
    Input 6: tới đây vào một buổi chiều mưa lạnh giá quán được trang trí bằng gam màu vàng ấm áp đồ ăn cũng khá đa dạng ở đây có món sườn cừu nướng là đặc sản của quán bình thường thì mình rất ghét thịt cừu vì mùi tanh của nó nhưng ở đây dường như mình không thấy mùi tanh cố hữu nêm nếm đậm đà trời mưa ăn sườn cừu nướng cảm giác hạnh phúc vô cùng ngoài ra quán có nhiều loại bánh ngọt lắm trang trí đẹp bánh mềm xốp xốp muốn tận chảy trong miệng luôn
    Output 6:
    #5
    tới đây vào một buổi chiều mưa lạnh giá quán được trang trí bằng gam màu vàng ấm áp đồ ăn cũng khá đa dạng ở đây có món sườn cừu nướng là đặc sản của quán bình thường thì mình rất ghét thịt cừu vì mùi tanh của nó nhưng ở đây dường như mình không thấy mùi tanh cố hữu nêm nếm đậm đà trời mưa ăn sườn cừu nướng cảm giác hạnh phúc vô cùng ngoài ra quán có nhiều loại bánh ngọt lắm trang trí đẹp bánh mềm xốp xốp muốn tận chảy trong miệng luôn
    {FOOD, positive}, {AMBIENCE, positive}

    As you can see, the output is in the order FOOD, AMBIENCE, SERVICE, PRICE. And in the format: {aspect, sentiment}. And for some reviews, there are no aspect that can be identify or the sentence has no meaning like example 5, for these reviews, I want you to remove it out of the output.
    For example, if I want you to do the task with these reviews: 

    ăn rất ngon được phục vụ chu đáo với 2 cô chú người huế có rất nhiều món như bò mọc chân giò chả cua rất đặc biết các bạn nên thử thập cẩm là đầy đủ tất cả một bát chỉ có 30 nghìn đồng 
    quán này khá đông khách vào buổi tối nên phải đi sớm mới có chỗ được v món ăn ngon giá cả mềm phục vụ chắc do quán đông nên khá thờ ơ tý nhưng bỏ qua mấy điểm đó thì quán ngon lành d
    nếu như muốn ăn các món đặc sản của miền bắc việt nam thì đây là địa điểm mà các bạn nên lưu lại menu phong phú giá cả bình dân muốn chắc chắn có bàn để thưởng thức thì các bạn sử dụng dịch vụ tablenow để book bàn trước nhé ngoài ra có thể đặt giao hàng tận nhà quá tiện lợi mà giá cả không thay đổi
    nha hang phuc vu cac mon an rat ngon nha
    nhiều ngày qua tình hình thế giới vô cùng căng thẳng vì những xung đột quân sự giữa mỹ và iran nhiều người lo lắng tình hình sẽ xấu đi và dẫn đến chiến tranh thế giới thứ 3
    tới đây vào một buổi chiều mưa lạnh giá quán được trang trí bằng gam màu vàng ấm áp đồ ăn cũng khá đa dạng ở đây có món sườn cừu nướng là đặc sản của quán bình thường thì mình rất ghét thịt cừu vì mùi tanh của nó nhưng ở đây dường như mình không thấy mùi tanh cố hữu nêm nếm đậm đà trời mưa ăn sườn cừu nướng cảm giác hạnh phúc vô cùng ngoài ra quán có nhiều loại bánh ngọt lắm trang trí đẹp bánh mềm xốp xốp muốn tận chảy trong miệng luôn

    The output should be: 
    #1
    ăn rất ngon được phục vụ chu đáo với 2 cô chú người huế có rất nhiều món như bò mọc chân giò chả cua rất đặc biết các bạn nên thử thập cẩm là đầy đủ tất cả một bát chỉ có 30 nghìn đồng 
    {FOOD,positive}, {SERVICE, positive}, {PRICE, positive}

    #2
    quán này khá đông khách vào buổi tối nên phải đi sớm mới có chỗ được v món ăn ngon giá cả mềm phục vụ chắc do quán đông nên khá thờ ơ tý nhưng bỏ qua mấy điểm đó thì quán ngon lành d
    {FOOD, positive}, {AMBIENCE, negative}, {SERVICE: negative}, {PRICE, positive}

    #3
    nếu như muốn ăn các món đặc sản của miền bắc việt nam thì đây là địa điểm mà các bạn nên lưu lại menu phong phú giá cả bình dân muốn chắc chắn có bàn để thưởng thức thì các bạn sử dụng dịch vụ tablenow để book bàn trước nhé ngoài ra có thể đặt giao hàng tận nhà quá tiện lợi mà giá cả không thay đổi
    {PRICE, positive}, {SERVICE, positive}

    #4
    nha hang phuc vu cac mon an rat ngon nha
    {FOOD, positive}

    #5
    tới đây vào một buổi chiều mưa lạnh giá quán được trang trí bằng gam màu vàng ấm áp đồ ăn cũng khá đa dạng ở đây có món sườn cừu nướng là đặc sản của quán bình thường thì mình rất ghét thịt cừu vì mùi tanh của nó nhưng ở đây dường như mình không thấy mùi tanh cố hữu nêm nếm đậm đà trời mưa ăn sườn cừu nướng cảm giác hạnh phúc vô cùng ngoài ra quán có nhiều loại bánh ngọt lắm trang trí đẹp bánh mềm xốp xốp muốn tận chảy trong miệng luôn
    {FOOD, positive}, {AMBIENCE, positive}.

    As you can see, the review number 5 has no meaning and be removed, then the review number 6 was assigned as the review number 5. Note that for each review, the output is in the format:
    #Review_no
    original_review
    {aspect, sentiment} pair(s)

    #next
    it has the space between 2 outputs. And {aspect, sentiment} pairs are seperated by ", "

    Note that for those reviews which no aspect (food, ambience, service, price) that can be identified. Ignore it and the next review (if there's aspect) will has that review number.
    Note that I just need the output which has negative, neutral sentiment. It mean that the output need contains at least 1 {aspect, sentimen} pairs with negative or neutral sentiment, not removing other the positive {aspect, sentiment} pairs in that output.
    For example: 
    Theses output should be kept: 
    #8096
    quán ăn bên dưới nhà khách không gian nhỏ nhắn phân thành 2 khu vực ăn uống và cafe món ăn làm hơi lâu vị tạm được không ngon không dở được cái giá cả phải chăng một phần thức ăn dao động khoảng 25 40 nghìn đồng  trà rất thơm
    {FOOD, neutral}, {AMBIENCE, negative}, {SERVICE, negative}, {PRICE, positive}
    -> because there is still at least an aspect that has negative sentiment. It's OK

    but this kind of output must be removed:
    #8095
    khách sạn và nhà hàng phục vụ đều rất tốt nhân viên phục vụ thân thiện giá cả cũng ok
    {SERVICE, positive}, {PRICE, positive}
    - because All the aspects have positive sentiment. It is not ok
    Note that the {aspect, sentiment} pair(s) must be unique in each output. 
    """
    return prompt

def read_review_file(prompt):
    original = prompt
    df = pd.read_csv("./200k_comments.csv")
    comments = df['comments']
    start = 60000
    end = 65000
    while start + 100 <= end:
        for comment in comments[start:start+100]:
            prompt += f"\n{comment}"
                
        response = model.generate_content(prompt)
        write_output("./data1.txt",response)
        prompt = original
        start += 100

def write_output(output_path, response):
    # Write the processed content to the output file
    with open(output_path, 'a', encoding='utf-8') as output_file:
        output_file.write(response.text)

prompt = get_train_prompt()
prompt +=  """Now, I want you to do the same for this review (note that I donnot need explaination) and the correct format"""
read_review_file(prompt)



