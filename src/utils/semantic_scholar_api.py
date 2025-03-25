"""
    Example output:
        {'total': 37673, 'offset': 0, 'next': 2, 'data': [{'paperId': '843831802a1d2d838d7874c4dea6ae8516112666', 'url': 'https://www.semanticscholar.org/paper/843831802a1d2d838d7874c4dea6ae8516112666', 'title': 'Machine Learning Supported Nano-Router Localization in WNSNs', 'abstract': 'Sensing data from the environment is a basic process for the nano-sensors on the network. This sensitive data need to be transmitted to the base station for data processing. In Wireless Nano-Sensor Networks (WNSNs), nano-routers undertake the task of gathering data from the nano-sensors and transmitting it to the nano-gateways. When the number of nano-routers is not enough on the network, the data need to be transmitted by multi-hop routing. Therefore, there should be more nano-routers placed on the network for efficient direct data transmission to avoid multi-hop routing problems such as high energy consumption and network traffic. In this paper, a machine learning-supported nano-router localization algorithm for WNSNs is proposed. The algorithm aims to predict the number of required nano-routers depending on the network size for the maximum node coverage in order to ensure direct data transmission by estimating the best virtual coordinates of these nano-routers. According to the results, the proposed algorithm successfully places required nano-routers to the best virtual coordinates on the network which increases the node coverage by up to 98.03% on average and provides high accuracy for efficient direct data transmission.', 'year': 2023, 'citationCount': 5, 'publicationTypes': ['JournalArticle'], 'authors': [{'authorId': '91889835', 'name': 'Ö. Güleç'}]}, {'paperId': 'eec35807668d5e2414d3021b7fd62b649ab25084', 'url': 'https://www.semanticscholar.org/paper/eec35807668d5e2414d3021b7fd62b649ab25084', 'title': 'An Improved Machine Learning Algorithm for Silver Nanoparticle Images: A Study on Computational Nano-Materials', 'abstract': 'Objectives : To study the determination and classiﬁcation of Scanning Electron Microscopy (SEM) images of silver nanoparticles using digital image processing techniques and classifying the images using various machine learning classiﬁers namely; SVM, K-NN, and PNN classiﬁers. Methods: Segmentation techniques namely; Fuzzy C-Means (FCM) and K-Means were applied to extract geometric features from SEM images of silver nanoparticles. The size of nanoparticles was determined and classiﬁed based on various nano applications. The categorization of silver nanoparticles was done based on the geometrical feature value, i.e', 'year': 2023, 'citationCount': 0, 'publicationTypes': ['JournalArticle'], 'authors': [{'authorId': '3106431', 'name': 'Parashuram Bannigidad'}, {'authorId': '2128955278', 'name': 'Namita Potraj'}, {'authorId': '12144814', 'name': 'P. M. Gurubasavaraj'}]}]}
    """

import requests

S2_API_KEY = 'YOUR API KEY'

def get_semantic_scholar_data(query, api_key, offset=0, limit=10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": api_key} if api_key else {}
    params = {
        "query": query,
        "offset": offset,
        "limit": limit,
        "fields": "title,abstract,year,citationCount,authors,url,publicationTypes"  # you can change this fields.
    }

    #time.sleep(2) # due to the rate limit:

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()      # Json format, as shown in the global comment `Example output`.
    except requests.RequestException as e:
        print(f"Error of RequestException: {str(e)}")
        return None


if __name__ == '__main__':

    query = "nano science, machine learning"  # a string that you want to query, keywords fashion works better than sentences.

    results = get_semantic_scholar_data(query=query, limit=2, api_key=S2_API_KEY)

    print(results)
