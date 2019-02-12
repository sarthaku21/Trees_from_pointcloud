extractTrees :
	g++ extractTrees.cpp `pkg-config opencv --cflags --libs` -o extractTrees
clean:
	rm extractTrees 
