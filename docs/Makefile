# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean livehtml autodoc

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets
clean:
	rm -rf $(BUILDDIR)/*
	@echo "Cleaned build directory."

livehtml:
	sphinx-autobuild -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) $(O)

autodoc:
	sphinx-apidoc -f -o . ../ridge_inference
	@echo "Generated API documentation."

# Add custom styles
custom_css:
	@mkdir -p _static
	@echo "/* Custom CSS for RidgeInference docs */\n\
	\n\
	/* Improve code block styling */\n\
	div.highlight {\n\
	    background-color: #f8f8f8;\n\
	    border-radius: 5px;\n\
	}\n\
	\n\
	/* Improve table styling */\n\
	table.docutils {\n\
	    border: 1px solid #e1e4e5;\n\
	    border-collapse: collapse;\n\
	    margin-bottom: 24px;\n\
	}\n\
	\n\
	table.docutils td, table.docutils th {\n\
	    padding: 8px 16px;\n\
	    border: 1px solid #e1e4e5;\n\
	}\n\
	\n\
	table.docutils th {\n\
	    background-color: #f3f6f6;\n\
	}\n\
	\n\
	/* Improve section headers */\n\
	h1, h2, h3, h4, h5, h6 {\n\
	    font-weight: 400;\n\
	    color: #2980B9;\n\
	}\n\
	\n\
	/* Improve API method display */\n\
	dl.method, dl.function {\n\
	    padding: 10px;\n\
	    border-left: 3px solid #6ab0de;\n\
	    background-color: #f8f8f8;\n\
	    margin-bottom: 20px;\n\
	}\n\
	\n\
	dl.method dt, dl.function dt {\n\
	    font-weight: bold;\n\
	    background-color: #e7f2fa;\n\
	    padding: 5px;\n\
	    border-radius: 3px;\n\
	}\n\
	" > _static/custom.css
	@echo "Created custom CSS."

# Build with all features
full: autodoc custom_css html
	@echo "Documentation built successfully."
