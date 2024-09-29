import requests
import time

# Replace with your actual website URL
url = "https://petmoo.com/wp-json/wp/v2/posts"

# Parameters to retrieve more posts and handle pagination
params = {
    'per_page': 100,  # Number of posts to retrieve per request (adjust as needed)
    'page': 1         # Start with the first page
}

all_posts = []
max_retries = 3  # Maximum number of retries for a failed request

while True:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)
            
            # Print debug information
            print(f"Debug: Response content for page {params['page']}:\n", response.text[:500])

            posts = response.json()

            if not posts:
                print("No more posts to fetch or empty response.")
                break

            all_posts.extend(posts)
            params['page'] += 1  # Go to the next page

            # Delay to avoid hitting server rate limits or overloading the server
            time.sleep(2)
            break  # Exit retry loop if successful

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            if response.status_code == 500 and attempt < max_retries - 1:
                print("Retrying after 5 seconds...")
                time.sleep(5)  # Wait before retrying
            else:
                break  # Exit retry loop if retries are exhausted
        except requests.exceptions.RequestException as err:
            print(f"Request error occurred: {err}")
            break

    if response.status_code != 200 or not posts:
        break  # Exit main loop if no more posts or critical error

# Process and save the data if posts were successfully fetched
if all_posts:
    import json
    with open('wordpress_posts.json', 'w') as f:
        json.dump(all_posts, f, indent=4)
    print(f"Extracted {len(all_posts)} posts.")
else:
    print("No posts were extracted.")
