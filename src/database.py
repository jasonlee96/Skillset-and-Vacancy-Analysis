from pymongo import MongoClient
import pandas as pd


# Database Object with Singleton Design Pattern
class Database(object):
    __instance = None
    __connection_string = "mongodb+srv://jason:jasondb@fyp-9p31t.mongodb.net/test?retryWrites=true&w=majority"
    __db = "fyp"
    __collection = "job_adverts"
    cluster = None
    db = None
    collection = None

    def __init__(self):
        if Database.__instance is None:
            Database.__instance = self
        else:
            raise Exception("Singleton Object cannot be initialized twice")

    @staticmethod
    def get_instance():
        if Database.__instance is None:
            Database()
        return Database.__instance

    # open the collection of the database
    def open(self):
        self.cluster = MongoClient(self.__connection_string)
        self.db = self.cluster[self.__db]
        self.collection = self.db[self.__collection]

    # Insert a list of job adverts into the database
    def insert_many(self, arr):
        self.collection.insert_many(arr)

    # Find all job adverts for particular key
    def find(self, show_id=True):
        field = None
        if not show_id:
            field = {"_id": 0}
        if field is None:
            df = pd.DataFrame(list(self.collection.find({})))
        else:
            df = pd.DataFrame(list(self.collection.find({}, field)))
        return df

    # TODO: Update one job adverts
    def update_with_id(self, _id, field):
        self.collection.update_one({"_id": _id}, {'$set': field}, upsert=False)

    # TODO: Delete job adverts for particular key

    # Close the database
    def close(self):
        Database.__instance = None
        self.cluster = None
        self.db = None
        self.collection = None
