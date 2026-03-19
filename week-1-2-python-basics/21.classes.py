class Student:
    """ """
    def __init__(self, name, major, gpa, is_on_probation):
        """

        :param name: param major:
        :param gpa: param is_on_probation:
        :param major: param is_on_probation:
        :param is_on_probation: 

        """
        self.name = name
        self.major = major
        self.gpa = gpa
        self.is_on_probation = is_on_probation

student1 = Student("Jim", "Business", 3.1,False)
student2 = Student ("Pam", "Art", 2.5, True)
print(student2.gpa)
