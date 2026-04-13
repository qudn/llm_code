def quicksort(arr, lo, hi):
    if lo >= hi:
        return 
    pivot_index = partition(arr, lo, hi)
    quicksort(arr, lo, pivot_index-1)
    quicksort(arr, pivot_index+1 hi)
def partition(arr, lo, hi):
    mid = (lo + hi)//2
    if arr[lo] > arr[mid]: arr[lo], arr[mid] = arr[mid], arr[lo]
    if arr[lo] > arr[hi]: arr[lo], arr[hi] = arr[hi], arr[lo]
    if arr[mid] > arr[hi]: arr[mid], arr[hi] = arr[hi], arr[mid]
    arr[mid], arr[hi-1] = arr[hi-1], arr[mid]
    pivot = arr[hi-1]

    i, j= lo, hi-1
    while True:
        i+=1
        while arr[i]<pivot: i+=1
        j-=1
        while arr[j]>pivot: j-=1
        if i>=j:
            break
        arr[i], arr[j] = arr[j], arr[i]
    arr[i], arr[hi-1] = arr[hi-1], arr[i]
    return i
