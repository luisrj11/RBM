# Pakages
import os
import pathlib

'''
===============================================================================
-----------------
Save the output :
-----------------

Crear where you goingo to save the output
===============================================================================
'''
class SaveOutput():
    
    def __init__(self,Name_Save_Output ) -> None:
        
        if type(Name_Save_Output) == str:
            # Where to save the figures and data files (automatically save in the current working directory)
            self.Path_Save_output = str(pathlib.Path().absolute())                      # Current working directory

            # Name where will save the output 
            self.Name_Save_Output = Name_Save_Output
        else:
            print('Name_Save_Output has to be a string type')     
                                            
    
    # Save figure  
    def figure_path(self,Name_save_figure):

        # Root where figures going to be saved
        PROJECT_ROOT_DIR = self.Path_Save_output +"/" + self.Name_Save_Output
        FIGURE_ID = PROJECT_ROOT_DIR + "/FigureFiles"                                   # Name the file 
        if not os.path.exists(PROJECT_ROOT_DIR):
            os.mkdir(PROJECT_ROOT_DIR)  

        if not os.path.exists(FIGURE_ID):
            os.mkdir(FIGURE_ID)

        return os.path.join(FIGURE_ID,Name_save_figure)
    
    # save data
    def data_path(self,Name_save_data):

        # Root where datas going to be saved
        PROJECT_ROOT_DIR = self.Path_Save_output +"/" + self.Name_Save_Output
        DATA_ID = PROJECT_ROOT_DIR + "/DataFile"                                        # Name the file 
        if not os.path.exists(PROJECT_ROOT_DIR):
            os.mkdir(PROJECT_ROOT_DIR)

        if not os.path.exists(DATA_ID):
            os.mkdir(DATA_ID)

        Name_save_data_ID = os.path.join(DATA_ID,Name_save_data)
        #if not os.path.exists(Name_save_data_ID):
        #    Creat_text = open((Name_save_data_ID),'w')  
        #    Creat_text.close()

        return Name_save_data_ID