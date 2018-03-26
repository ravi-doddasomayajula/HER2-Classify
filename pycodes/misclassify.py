false_positive_mononuclear = np.intersect1d(np.where(y_pred == 1), np.where(y_test == 0))

img = X_test[false_positive_mononuclear[0]]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


