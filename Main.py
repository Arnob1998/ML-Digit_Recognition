import traceback
import DrawDigit
import ModelTraining
import DigitManagement
from matplotlib import pyplot as plt
import os
import shutil

if __name__ == "__main__":

    def plot_digit(data, x=28, y=28):
        image = data.reshape(x, y)
        plt.imshow(image, cmap=plt.cm.binary, interpolation="nearest")
        plt.axis("on")
        plt.show()

    def manualModelSelection():
        model_list = os.listdir(os.getcwd() + "\\saved_model")
        print("\nModels available : \n")
        for i in range(len(model_list)):
            print(str(i) + ". " + model_list[i])

        selection = input("\nselect : ")
        model_name = model_list[int(selection)]
        return model_name


    def saveCustomData(predicted, dir=os.getcwd() + "\\custom_data"):
        list_dir = os.listdir(dir)
        pred = None
        pred_type = None

        print("Was the prediction correct?")
        print("1.Yes\t2.No")
        choice = int(input("Choice : "))
        if choice == 1:
            pred = predicted
            pred_type = "TP"
        elif choice == 2:
            pred = int(input("Enter the correct label : "))
            pred_type = "FP"

        prednum_lis = []
        for file in list_dir:
            if str(pred) + "_" in file:
                prednum_lis.append(file)

        if prednum_lis == []:
            prednum_lis.append(str(pred) + "_0-" + pred_type)

        last_digit = []
        for full_name in prednum_lis:
            underscore_index = full_name.index('_')
            last_digW_ext = full_name[underscore_index + 1:]

            dot_index = last_digW_ext.index('-')
            only_digit = last_digW_ext[:dot_index]
            last_digit.append(only_digit)

        shutil.copyfile(os.getcwd() + "\\screenshot.jpeg",
                        os.getcwd() + "\\custom_data\\" + str(pred) + "_" + str(
                            int(max(last_digit)) + 1) + "-" + pred_type + ".jpeg")

    try:
        # manual_model = manualModelSelection()
        manual_model = "knnMINST_N1_grayscale"

        drawObj = DrawDigit.DrawDigitClass(shape_width=40, shape_height=40)
        drawObj.start()

        digManObj = DigitManagement.DigitTransformClass(anti_aliasing=True) # anti_aliasing = false for binary digit
        trans_digit = digManObj.transformPipeline()                         # anti_aliasing = Ture for grayscale digit

        modelObj = ModelTraining.SelectModelClass()

#         model = modelObj.load_model(manual_model)
        # model = modelObj.load_ensemble()
        model = modelObj.train_ANN()

        # print("Log : Plotting drawn image")
        # plot_digit(trans_digit)

        # pred = model.predict([trans_digit])

        pred = model.predict_classes(trans_digit.reshape(1,-1)) #only for ANN
        print("Prediction : " + str(pred))

        print() # todo add predict proba
        # saveCustomData(pred[0])

    except Exception as err:
        traceback.print_exc()

# MODEL EVALUATION
# When binary trained model : knnMINST_N1_grayscale is used | DigitTransformClass(anti_aliasing=False) -> grayscale drawn digit:

# recommended : draw as large as possible and slowly
# Good : 0,1,2,3
# Bad : 5,6,7,8(perfect-slowly drawn works),9
