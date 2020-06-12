from question1 import linearRegressionLoader, regularisationLoader, bestFitLineLoader
from question2a import logisticRegressionLoader
from question2bc import mnistLoader

def getCorrectInput(x):
    validOption = False
    while(not validOption):
        try:
            inp = int(input())
            if inp in range(1, x+1):
                validOption = True
                return inp    
            else:
                print("Please enter a valid option(1-" + str(x) + "):", end =" ")
        except:
            print("Please enter a valid option(1-" + str(x) + "):", end =" ")

def mainMenu(useSavedModels):
    print("-" * 10 + "" + "-" * 10)    
    print("Main Menu")
    print("1. Question 1")
    print("2. Question 2")
    print("3. Exit")
    print("Choose option(1-3):", end =" ")
    option = getCorrectInput(3)
    if(option == 1):
        question1(useSavedModels)
    elif(option == 2):
        question2(useSavedModels)
    elif(option == 3):
        exit()

def question1(useSavedModels):
    print("-" * 10 + "" + "-" * 10)    
    print("Question 1 Menu")
    print("1. Linear Regression")
    print("2. Regularisation")
    print("3. Best Fit Line")
    print("4. Go Back")
    print("5. Exit")
    print("Choose option(1-5):", end =" ")
    option = getCorrectInput(5)
    if(option == 1):
        linearRegressionLoader(useSavedModels)
        question1(useSavedModels)
    elif(option == 2):
        regularisationLoader(useSavedModels)
        question1(useSavedModels)
    elif(option == 3):
        bestFitLineLoader(useSavedModels)
        question1(useSavedModels)
    elif(option == 4):
        mainMenu(useSavedModels)
    elif(option == 5):
        exit()

def question2(useSavedModels):
    print("-" * 10 + "" + "-" * 10)    
    print("Question 2 Menu")
    print("1. Logistic Regression")
    print("2. MNIST")
    print("3. Go Back")
    print("4. Exit")
    print("Choose option(1-4):", end =" ")
    option = getCorrectInput(4)
    if(option == 1):
        logisticRegressionLoader(useSavedModels)
        question2(useSavedModels)
    elif(option == 2):
        mnistLoader(useSavedModels)
        question2(useSavedModels)
    elif(option == 3):
        mainMenu(useSavedModels)
    elif(option == 4):
        exit()