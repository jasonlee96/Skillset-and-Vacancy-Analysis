# Job Data Model
class Job:
    def __init__(self, title, href):
        self.title = title
        self.href = href
        self.description = None

    def add_description(self, desc):
        self.description = desc

    # toString Method()
    def __str__(self):
        print(self.title, self.href, self.description, sep=" ")

    # Convert to dict format that helps in inserting into database
    def to_dict(self):
        return {"title": self.title, "href": self.href, "description": self.description}
