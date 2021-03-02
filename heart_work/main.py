# coding=utf-8
import os, sys, time, threading as td
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
from dicom2jpg import preprocess

p = None

def start(inputFolderTextField, outputFolderTextField, p_noCombobox, stateStringVar, startButton, isOnlyUSImageCheckbutton, isOnlyUSImageCheckVar):
    inputFolder = inputFolderTextField.get()
    outputFolder = outputFolderTextField.get()
    p_no = int(p_noCombobox.get())

    if not os.path.exists(inputFolder):
        tk.messagebox.showerror("Error", "輸入圖片路徑不存在!!!")
        return

    if not os.path.exists(outputFolder):
        tk.messagebox.showerror("Error", "輸出圖片路徑不存在!!!")
        return

    isMultiprocessing=True
    if p_no==0 or p_no==1:
        isMultiprocessing=False

    inputFolderTextField['state'] = "disabled"
    outputFolderTextField['state'] = "disabled"
    p_noCombobox['state'] = "disabled"
    startButton['state'] = "disabled"
    isOnlyUSImageCheckbutton['state'] = "disabled"

    subtd = td.Thread(target=preprocess, args=(inputFolder, outputFolder, 
        isMultiprocessing, p_no, isOnlyUSImageCheckVar.get()) )
    subtd.start()

    with open("state.txt", "w") as f:
        f.write("1")
    
    td.Thread(target=checkDone, args=(stateStringVar, inputFolderTextField, outputFolderTextField, p_noCombobox, startButton, isOnlyUSImageCheckbutton)).start()

def checkDone(stateStringVar, inputFolderTextField, outputFolderTextField, p_noCombobox, startButton, isOnlyUSImageCheckbutton):
    startTime = time.time()
    while True:
        time.sleep(5)
        stateStringVar.set("Processing!!!")
        stateStringVar.set("Processing!!!")
        f = open("state.txt", "r")
        state = f.readline().strip()
        f.close()
        if state == "0":
            stateStringVar.set("Done!!!")
            inputFolderTextField['state'] = "normal"
            outputFolderTextField['state'] = "normal"
            p_noCombobox['state'] = "normal"
            startButton['state'] = "normal"
            isOnlyUSImageCheckbutton['state'] = 'normal'
            tk.messagebox.showinfo("DONE", "Taking Time: %.2f" %(time.time() - startTime) + " seconds.")
            os.remove("state.txt")
            break

def browseFolder(window, stringVar):
    '''
    Parameters:

    Return:

    Description:
        The function called by browse folder button is to browse the error output folder.
    '''
    from tkinter import filedialog
    filePath = filedialog.askdirectory(initialdir="./")
    stringVar.set(filePath)

def on_closing():
    '''
    Parameters:

    Return:

    Description:
        Called when the window closing.
    '''
    sys.exit() 

def main():
    window = tk.Tk()
    window.title("DICOM 2 JPG bot")
    window.geometry("800x200")
    window.configure(background="white")

    # Visible Components settings
    text1 = tk.Label(window, text="請輸入DICOM資料夾路徑：", font=("Times", "20"), bg="white")
    text1.place(x=20, y=20)

    inputFolderStringVar = tk.StringVar()
    inputFolderTextField = tk.Entry(window, show=None, textvariable=inputFolderStringVar, font=("Times", "16"))
    inputFolderTextField.place(x=370, y=24, width=360)
    inputFolderStringVar.set("")

    browseInputFolderButton = tk.Button(window, text="...", font=("Times", "14"), command=lambda: browseFolder(window, inputFolderStringVar))
    browseInputFolderButton.place(x=740, y=20, width=30)

    text2 = tk.Label(window, text="請輸入JPG輸出資料夾路徑：", font=("Times", "20"), bg="white")
    text2.place(x=20, y=60)

    outputFolderStringVar = tk.StringVar()
    outputFolderTextField = tk.Entry(window, show=None, textvariable=outputFolderStringVar, font=("Times", "16"))
    outputFolderTextField.place(x=370, y=65, width=360)
    outputFolderStringVar.set("")

    browseOutputFolderButton = tk.Button(window, text="...", font=("Times", "14"), command=lambda: browseFolder(window, outputFolderStringVar))
    browseOutputFolderButton.place(x=740, y=65, width=30)

    text3 = tk.Label(window, text="請選擇處理器數量：", font=("Times", "20"), bg="white")
    text3.place(x=20, y=100)

    p_noCombobox = ttk.Combobox(window, state="readonly", values=[i for i in range(9)], textvariable=False)
    p_noCombobox.grid(column=0, row=1)
    p_noCombobox.place(x=270, y=110, width=50)
    p_noCombobox.current(1)

    isOnlyUSImageCheckVar = tk.IntVar()
    isOnlyUSImageCheckbutton = tk.Checkbutton(window, text="是否只留US圖像", variable=isOnlyUSImageCheckVar, onvalue=1, offvalue=0, bg='white', font=("Times", "14"))
    isOnlyUSImageCheckbutton.place(x=370, y=100)

    stateStringVar = tk.StringVar()
    stateText = tk.Label(window, textvariable=stateStringVar, font=("Times", "20", "bold"), bg="white", fg="red")
    stateText.place(x=20, y=140)
    stateStringVar.set("注意：可以接受相對或絕對路徑，但是路徑中不能有中文。")

    startButton = tk.Button(window, text="Start!!", font=("Times", "14"), 
        command= lambda : start(inputFolderTextField, outputFolderTextField, p_noCombobox, stateStringVar, 
        startButton, isOnlyUSImageCheckbutton, isOnlyUSImageCheckVar))
    startButton.place(x=600, y=100)

    # Keep fresh the window
    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()

if __name__ == "__main__":
    main()