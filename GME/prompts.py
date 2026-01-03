FACT_EXTRACTION_PROMPT = """
You are a professional AI expert tasked with distilling a conversation into core memories.
Your primary goal is to capture information that, if forgotten, would make future conversations feel disjointed, impersonal, or lacking in life details. Extract 1-4 pieces of information as appropriate.
[Context: Overall User Summary]
{global_summary}

[New Conversation Content]
"{profile_text}"

[Guiding Principles]
1.  **Be Specific**: Extract precise facts. For example, extract "User likes jazz music," not "User likes music."
2.  **Focus on Key Information**: Pay attention to preferences (likes/dislikes), goals, plans, personal experiences, habits, states, relationships, and key facts mentioned.
3.  **Avoid Chitchat**: Do not extract generic greetings (e.g., "hello," "how are you?") or conversation fillers that lack meaningful content.
4.  **Concise and Standalone**: Each memory must be a concise and standalone statement.
5.  **Third-Person Perspective**: All memories must be written in third person (e.g., "User plans to...").
6.  **Avoid Duplication**: If the new conversation content only repeats information already covered in the summary, extract an empty list.

[Output Format]
You must respond with a valid JSON object containing a single key "facts," where the value should be a list of strings.

[Example]
User Summary: "The user is a software developer."
New Conversation Content: "I finally booked my flight to Tokyo for next month's vacation! I'm so excited to try authentic ramen."
Your Output:
{{
  "facts": [
    "User booked a flight to Tokyo for a vacation next month.",
    "User is excited to try authentic ramen in Tokyo."
  ]
}}

Now, based on the provided summary and new conversation content, extract the core memories.
"""
MEMORY_UPDATE_PROMPT = """
You are a meticulous memory manager for an AI assistant.
Your task is to analyze a new piece of information against a list of existing similar memories and decide on a precise action to maintain the consistency and accuracy of the knowledge base.
Your comparison is not strictly based on the category, but rather on whether the facts overlap.

**Action Guidelines:**
- **ADD**: Choose this if the new information is entirely new and unrelated or not on the same level as any existing memory, with no overlap or contradiction.
- **UPDATE**: Choose this if the new information relates to an existing memory and adds crucial details, clarifies, or corrects it. The goal is to merge the new information and the most relevant existing memory into a single, more accurate memory.
- **DELETE**: Choose this only if the new information directly contradicts and invalidates an existing memory (e.g., a plan was canceled, or a preference has changed).
- **NOOP**: Choose this if the new information is an exact repetition, highly redundant to an existing memory, or trivial conversational filler, and provide a reason.

**Existing Similar Memories:**
{existing_memories}

**New Information:**
{new_fact}

---
**Output Format Rules:**

*   If you choose **UPDATE**:
    - Line 1: UPDATE
    - Line 2: Write a single, concise, updated memory sentence that merges the new information with the most relevant old fact.

*   If you choose **DELETE**:
    - Line 1: DELETE
    - Line 2: Provide a brief, one-sentence reason for the deletion.

*   If you choose **ADD**:
    - Line 1: Simply output the single word ADD.

*   If you choose **NOOP**:
    - Line 1: NOOP
    - Line 2: Provide a brief reason explaining why no action is taken.

---
**Examples:**

*Example 1 (Update)*
Existing Memories:
- User is planning to learn a new language.
New Information: User has decided to learn Rust.
Your Output:
UPDATE
User decided to learn the Rust programming language.

*Example 2 (Delete)*
Existing Memories:
- User has an important project presentation next Monday.
New Information: The user's project presentation on Monday was a success.
Your Output:
DELETE
The event has already passed and was successful.

*Example 3 (Add)*
Existing Memories:
- User likes sci-fi novels.
New Information: User's cat is named 'Pidan'.
Your Output:
ADD

*Example 4 (No-Op)*
Existing Memories:
- User enjoys watching sci-fi movies like 'Dune 2'.
New Information: The user thought 'Dune 2' was visually stunning.
Your Output:
NOOP
The new information is essentially the same as existing memory.

---
Now, provide your decision for the new information below:
"""
SUMMARY_UPDATE_PROMPT = """
You are a professional user profiling expert. Your task is to generate an updated and more comprehensive user summary based on an existing summary and a recent raw conversation.

[Current User Summary]
{current_summary}

[Recent Raw Conversation]
{recent_raw_texts}

[Updating Requirements]
1.  Read through the "Recent Raw Conversation" and extract relevant information.
2.  Integrate these new details naturally into the "Current User Summary," incorporating them into a cohesive and optimized summary.
3.  Avoid simply listing the raw conversation as-is; instead, craft a fluid and descriptive summary.

[Output]
Provide the updated user summary in its entirety.
"""
FINAL_QA_PROMPT_JSON = """
You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions.

# Your task: 
Answer the user's current question in a personalized way by considering their past memories to understand their preferences.

# Your output: 
You should generate a valid json object in a ```json ``` block that contains the following fields:
- personalized_answer: contains the personalized answer to the user's current question.

your input:
[Retrieved Relevant User Memories]
- {retrieved_memories}
[User's Question]
{question}

[Answer]
"""
FINAL_QA_PROMPT1 = """
You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past retrieved relevant user memories.

# Your task: 
Answer the user's current question in a personalized way by considering their past posts to understand their preferences.The following retrieved user memories are potentially supplemental to the user's question; please carefully consider whether to integrate them into your response.  

your input:
[Retrieved Relevant User Memories]
- {retrieved_memories}
[User's Question]
{question}

[Answer]
"""

FINAL_QA_PROMPT="""
You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past retrieved relevant user memories.

# Your input:
- The user's current question from a post.
- The user's relevant past memories to learn about the user's preferences.

# Your task: 
Answer the user's current question in a personalized way by considering their past posts to understand their preferences.

# Your output:
Generate a direct, personalized answer to the user's current question.
your input:
[Retrieved Relevant User Memories]
- {retrieved_memories}
[User's Question]
{question}

[Answer]
"""


