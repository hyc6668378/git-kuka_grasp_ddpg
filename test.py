 
import threading
import time
 
lock = threading.Lock()
l = []
 
def test1(n):
	lock.acquire()
	l.append(n)
	print(l)
	lock.release()
 
def test(n):
	l.append(n)
	print(l)
 
def main():
	for i in range(0, 10):
		th = threading.Thread(target=test, args=(i, ))
		th.start()
if __name__ == '__main__':
	main()
