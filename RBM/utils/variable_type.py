'''
===============================================================================
Creat a different type alias
===============================================================================
'''
class Type:
    vector = type('vector', (), {})
    matrix = type('matrix',(),{})
    tensor = type('tensor',(),{})
    
# Know all valid attributes 
class Atributes:
     Atributes = dir

if __name__ == "__main__":
    print('===========================================')
    print('THIS IS A TEST, IT HAS NOT PHYSICS MEANING')
    print('===========================================')
    print(Atributes.Atributes(Type))