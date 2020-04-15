from extraction import Extraction
import time
from database import Database

key = "software engineer"
ext = Extraction('https://www.jobstreet.com.my/en/job-search/job-vacancy.php',
                 {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'},
                 key=key)


def extract():
    # Stopwatch to see the time taken for extraction
    start = time.time()
    data = ext.get_job_titles()
    stop = time.time()
    job_adverts = []
    for item in data:
        job_adverts.append(item.to_dict())

    print("size: ", len(job_adverts))
    print("time used for extraction: ", (stop - start))
    return job_adverts


# Driver Module
def main():
    # Extract data Driver
    data = extract()

    # MongoDb Initialization
    db = Database.get_instance()
    start = time.time()
    db.open(collection="mix_adverts")
    print("time used for db initialization: ", (time.time() - start))
    start = time.time()
    db.insert_many(data)
    print("time used for insertion: ", (time.time() - start))

    # start = time.time()
    # # cast the mongo data into pandas DataFrame
    # df = db.find(show_id=False)
    # print("time used for read: ", (time.time() - start))
    # print(df['description'])
    #
    # # Store into txt file and test the read speed and write speed for both (https://www.youtube.com/watch?v=irnj19jz8uI)
    # start = time.time()
    # # TODO: try to use pandas write into txt to check the performance
    # print("time for writing into txt file: ", (time.time() - start))
    #
    # print(df['new_description'])


if __name__ == "__main__":
    main()