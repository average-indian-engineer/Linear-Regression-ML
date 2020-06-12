# Sources
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

from menus import mainMenu
import sys

if __name__ == "__main__":
    useSavedModels = False   
    try:
        sys.argv[1:][0] == '-s'
        useSavedModels = True   
    except:
        useSavedModels = False   
    if(useSavedModels):
        print("-" * 10 + "" + "-" * 10)    
        print("SAVED MODELS WILL BE USED...")
    mainMenu(useSavedModels)