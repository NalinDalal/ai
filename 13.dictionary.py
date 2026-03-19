monthConversions = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
}

print(monthConversions["Mar"])

x={'pork':25.3, 'beef':33.8, 'chicken':22.7}
x= dict([('pork', 25.3), ('beef', 33.8), ('chicken', 22.7)])
x= dict(pork=25.3, beef=33.8, chicken=22.7)

x. keys ()  #returns list of keys in x
x. values () # returns list of values in x

x. items () # returns list of key-value tuple pairs in x

item in x. values ()# tests membership in x: returns boolean
