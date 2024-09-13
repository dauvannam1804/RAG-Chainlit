# RAG-With-Langchain

This repository contains the code for a Retrieval-Augmented Generation (RAG) system that leverages a large language model to answer questions based on PDF documents. The approach is inspired by insights shared by [AI VIETNAM](https://www.facebook.com/aivietnam.edu.vn/posts/778244334418287?rdid=T8Lv2BzNXM8If0u6). The project also utilizes Chainlit, a robust framework designed for building interactive conversational AI applications. Chainlit enables the creation of an intuitive user interface for the question-answering system, enhancing user experience.

## Features
- **Question Answering:** Retrieve and answer questions based on the content of PDF documents.
- **Interactive Interface:** Built using Chainlit for a user-friendly chat interface.
- **Versatile Deployment:** Supports both local and Google Colab environments.

## Getting Started

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/dauvannam1804/RAG-Chainlit.git
    ```

2. **Create a virtual environment:**
    ```sh
    conda create -n rag-chainlit python=3.9
    ```

3. **Activate the virtual environment:**
    ```sh
    conda activate rag-chainlit
    ```

4. **Run the application locally:**
    ```sh
    chainlit run src/app.py --host 0.0.0.0 --port 8000 &> ./logs.txt &
    ```

### Running on Google Colab

To run the project on Google Colab, follow the guide provided in this [notebook](https://colab.research.google.com/drive/1adw5dAjy4Idqd2xzhJvbLbiZnJSNNzGu?usp=drive_link).

1. **Running on Colab:**

   ![image](https://github.com/user-attachments/assets/d8af21b6-62df-47d8-b62f-a9c7bb7f5d6d)

2. **In-app chat interface:**

   ![image](https://github.com/user-attachments/assets/bb5755ae-aaa6-4bfc-a857-430de744681d)

3. **Test a chat on the finished interface:**

   ![image](https://github.com/user-attachments/assets/aa939026-7251-4848-b7ce-e831c4eea497)
   ![image](https://github.com/user-attachments/assets/b45cd7e1-1047-49cf-8423-fa2ddc013f8a)

## Additional Information

- **Chainlit Framework:** Provides a seamless way to develop and deploy conversational AI with a focus on creating interactive and user-friendly experiences.

For more details, please refer to the [documentation]([https://example.com/documentation](https://docs.chainlit.io/get-started/overview)) and the [Chainlit project page](https://github.com/Chainlit/cookbook).

