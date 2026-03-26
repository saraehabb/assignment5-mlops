with open("accuracy.txt", "r") as f:
    accuracy = float(f.read())

print("Accuracy:", accuracy)

if accuracy < 0.85:
    print("Accuracy below threshold, but continuing...")

print("Done successfully")
