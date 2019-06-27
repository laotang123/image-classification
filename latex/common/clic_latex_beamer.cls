% ---------------------------------------
%	Template Declaration
% ---------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{clic_latex_beamer}[2014/06/03 CLIC's Latex beamer template]

% ---------------------------------------
%	Load base class
% ---------------------------------------
% \LoadClass[handout,10pt]{beamer}
\LoadClass[10pt]{beamer}

% ---------------------------------------
%	Load required packages
% ---------------------------------------
%For French stuff
\RequirePackage[english]{babel}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
%Creative commons icons
\RequirePackage{ccicons}
%Embed LaTeX code
\RequirePackage{listings}
%Adds additional symbols such as \danger
%\RequirePackage{fourier}
%Adds additional colors
\RequirePackage{color}

% ---------------------------------------
%	Theme customization
% ---------------------------------------

%List of available themes and colors: http://www.hartwork.org/beamer-theme-matrix/
\usetheme{Copenhagen}
\usecolortheme{beaver}
%Some more customization
\setbeamercolor{item}{fg=red}

%Flat design for items bullets (cf. http://tex.stackexchange.com/questions/52123/beamer-increase-radius-of-circle-in-toc-and-include-subsection-bullet)
\setbeamertemplate{items}[circle]
\setbeamertemplate{section in toc}[circle]
\setbeamertemplate{subsection in toc}
{\leavevmode\leftskip=2em%
  \llap{%
    \usebeamerfont*{section number projected}%
    \usebeamercolor{section number projected}%
    \begin{pgfpicture}{-1ex}{0ex}{1ex}{1ex}
      \color{bg}
      \pgfpathcircle{\pgfpoint{0pt}{1ex}}{0.5ex}
      \pgfusepath{fill}
    \end{pgfpicture}\kern1.25ex%
  }%
  \inserttocsubsection\par}

%Display table of content for each section at the beginning of the section


%Options for listings package
\definecolor{darkgreen}{rgb}{0,0.6,0}
\lstset
{
    language=[LaTeX]TeX,
    breaklines=true,
    basicstyle=\tt\scriptsize,
    texcsstyle=*\color{blue},
    inputencoding=utf8,
    keywordstyle=\color{darkgreen},
    keywords={$},
	otherkeywords={$, \{, \}, \[, \]},
    commentstyle=\color{red},
    literate={é}{{\'e}}1 {è}{{\`e}}1 {à}{{\`a}}1 {ç}{{\c{c}}}1 {œ}{{\oe}}1 {ù}{{\`u}}1 {É}{{\'E}}1 {È}{{\`E}}1 {À}{{\`A}}1 {Ç}{{\c{C}}}1 {Œ}{{\OE}}1 {Ê}{{\^E}}1 {ê}{{\^e}}1 {î}{{\^i}}1 {ô}{{\^o}}1 {û}{{\^u}}1
}

%
%\setbeamercolor{alerted text}{fg=yellow}
