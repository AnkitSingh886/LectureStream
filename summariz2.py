from transformers import BartForConditionalGeneration, BartTokenizer
# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained(
    'facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained(
    'facebook/bart-large-cnn')
def summarize_text(text, maxSummarylength=500):
    # Encode the text and summarize_text
    inputs = tokenizer.encode("summarize_text: " +
                              text,
                              return_tensors="pt",
                              max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=maxSummarylength,
                                 min_length=int(maxSummarylength/5),
                                 length_penalty=10.0,
                                 num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def split_text_into_pieces(text,
                           max_tokens=900,
                           overlapPercent=10):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Calculate the overlap in tokens
    overlap_tokens = int(max_tokens * overlapPercent / 100)

    # Split the tokens into chunks of size
    # max_tokens with overlap
    pieces = [tokens[i:i + max_tokens]
              for i in range(0, len(tokens),
                             max_tokens - overlap_tokens)]

    # Convert the token pieces back into text
    text_pieces = [tokenizer.decode(
        tokenizer.convert_tokens_to_ids(piece),
        skip_special_tokens=True) for piece in pieces]

    return text_pieces
def recursive_summarize_text(text, max_length=200, recursionLevel=0):
    recursionLevel=recursionLevel+1
    print("######### Recursion level: ",
          recursionLevel,"\n\n######### ")
    tokens = tokenizer.tokenize(text)
    expectedCountOfChunks = len(tokens)/max_length
    max_length=int(len(tokens)/expectedCountOfChunks)+2

    # Break the text into pieces of max_length
    pieces = split_text_into_pieces(text, max_tokens=max_length)

    print("Number of pieces: ", len(pieces))
    # summarize_text each piece
    summaries=[]
    k=0
    for k in range(0, len(pieces)):
        piece=pieces[k]
        print("****************************************************")
        print("Piece:",(k+1)," out of ", len(pieces), "pieces")
        print(piece, "\n")
        summary =summarize_text(piece, maxSummarylength=max_length/3*2)
        print("SUMNMARY: ", summary)
        summaries.append(summary)
        print("****************************************************")

    concatenated_summary = ' '.join(summaries)

    tokens = tokenizer.tokenize(concatenated_summary)

    if len(tokens) > max_length:
        # If the concatenated_summary is too long, repeat the process
        print("############# GOING RECURSIVE ##############")
        return recursive_summarize_text(concatenated_summary,
                                   max_length=max_length,
                                   recursionLevel=recursionLevel)
    else:
      # Concatenate the summaries and summarize_text again
        final_summary=concatenated_summary
        if len(pieces)>1:
            final_summary = summarize_text(concatenated_summary,
                                  maxSummarylength=max_length)
        return final_summary
    # Example usage
text = '''A linked list is a list of structs or classes that include a pointer. This pointer points to the next struct or class in its list as it gets added. Normally we call these pointers next to give them a descriptive name as to what they're pointing to, but they can be anything. Unlike an array where you have to statically set the number of elements you need, a linked list is a great alternative because you can link them dynamically, allowing your list to grow or shrink as needed. To create a linked list, we first need to create a struct or a class. I'm going to use a struct. I've defined my struct as node. In my node I have a data point, and I also have a pointer, a pointer to another node. This line node link is a pointer to a node. The line after that typedef node is redefining what node pointer looks like. So instead of having to require an asterisk, anytime I use a pointer, I can now just declare them as type node pointers. It's the same thing. You can use typedef anywhere you need, but it's highly recommended to only use them when needed. So in my main I have node pointer head. This creates a new node pointer called head. Right now it's not pointing to anything, but we'll get to that in a minute. The next line says head equals new node. So now I'm going to point to a brand new node. And remember, every time I create a node, I get the same data points that are defined in the struct node. So I have an int data section which I've defined in blue, and then I have a node pointer right now, again, it's not pointing to anything. The next two lines populate my new node with data. We're also introducing the arrow operator. This is used when you're using pointers in reference to populating data. So here I have my pointer head pointing to a new node, and now I want to put 20 in the data spot. I'm also pointing the link portion to null. Next line is just for data debugging. If I try to print out the data that my head is pointing to, it should give me a 20. So all I need to say is head arrow data, and that will get the data that the head pointer is pointing to. But what if we have more data? What if we want to insert a whole bunch of data into our list? Here I've written an insert function. It requires a pointer passed by reference. Remember, pass by reference versus pass by value, and a data point, and it'll take that data point and populate a new node. So the first line in my function declares a new node pointer called temp pointer. I've named it temp ptr for short. All that does is create a new pointer. Currently it's not pointing to anything. The next line creates a new node and makes the temp pointer point to this new node. So now I have a brand new node. Anytime I declare a node, I get the same two pieces of data, an integer for the data type, and then a link for the next node. I populate my data. I put 30 from my parameter list into the data portion, and then the next line says temp pointer link. So now I'm linking the next node to head. So remember our head was pointing to a node with 20 in it. So now I have two pointers pointing to the node with 20. The last line redirects head to the front of my list. Head should always be the front of your list. It makes it easier to find the start of your list if it's an empty list or where to start searching and sorting from. So now my head is pointed back around the same place my temp pointer is pointed to, which is the node that has 30. As soon as I return back from my function, temp pointer will be deleted, because remember, it was only created in that function. So now my list only contains a head pointer pointing to the node with the 30, the node with the 30 pointing to the next node with the 20, and the node with the 20 pointing to null. We could keep inserting, but let's print out our list. So when I return back from my function, the next line creates a new temporary pointer, currently not pointing to anything. Then I set it to point to the same place as head. Head should be at the front of my list. And now I say wild temp, or that pointer is not equal to null. So while I'm not at the end of my list, I want to see out the data in the node that I'm pointing to and then move to the next node. So after it comes from head points to 30, it'll print 30, and then this line will move the temp pointer to the next node in the list. The next move will end up being null because the node with 20 in its link does point to null. So if I moved one more time, I would be looking at null and I would come out of my loop. But by then I should have printed 20 and 30 on my console. Let's take a look. So here I have copied everything down just the way we had it. If I run it, it prints 30, and finally 20. So it works. Although a linked list with only two pieces of data is not very useful. But you can see how you can insert and delete nodes as you need them.'''

final_summary = recursive_summarize_text(text)
print("\n%%%%%%%%%%%%%%%%%%%%%\n")
print("Final summary:", final_summary)