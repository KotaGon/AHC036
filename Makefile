COMPILER = g++
CFLAGS   = -g -Wall -O3 -std=c++20
LDFLAGS  = -lscip 
LIBS     = -L/usr/local/lib 
INCLUDE  = -I./include -I./ -I/usr/local/include
TARGET   = optimize
OBJDIR   = ./obj
INCDIR   = ./include 
SOURCES  = $(wildcard *.cc)
HEADERS = $(wildcard $(INCDIR)/*.h $(INCDIR)/*hpp)
OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cc=.o))

$(TARGET): $(OBJECTS) 
	$(COMPILER) -o $@ $^ $(LDFLAGS) $(LIBS)
	
$(OBJDIR)/%.o: %.cc
	@[ -d $(OBJDIR) ]
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $<

$(OBJECTS) : $(HEADERS)

all: clean $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)
