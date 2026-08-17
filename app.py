import string
import re
import numpy as np
from collections import Counter
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math
import threading
from pathlib import Path

# --- Configuration for Scoring ---
UNIGRAM_WEIGHT = 0.4
BIGRAM_WEIGHT = 0.6
BIGRAM_SMOOTHING_ALPHA = 1

class SpellCheckerModel:
    def __init__(self):
        self.word_probs = None
        self.bigram_counts = None
        self.first_word_counts = None
        self.vocab = None
        self.model_loaded = False
    
    def read_corpus(self, filename):
        """Read and process corpus file"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read().lower()
                words = re.findall(r'\w+', text)
            return words
        except FileNotFoundError:
            return None
    
    def build_language_model(self, corpus):
        """Build language model from corpus"""
        # Word frequencies
        word_counts = Counter(corpus)
        total_words = sum(word_counts.values())
        word_probs = {word: count/total_words for word, count in word_counts.items()}
        
        # Bigram frequencies for context
        bigrams = [(corpus[i], corpus[i+1]) for i in range(len(corpus)-1)]
        bigram_counts = Counter(bigrams)
        
        # Pre-calculate counts for denominators in bigram probabilities
        first_word_counts = Counter(w1 for w1, w2 in bigrams)
        
        return word_probs, bigram_counts, first_word_counts, set(corpus)
    
    def load_model(self):
        """Load the spell checker model"""
        # Try to locate big.txt
        corpus_file = Path("big.txt")
        if not corpus_file.exists():
            print("Warning: 'big.txt' not found. Spell checking will not work.")
            return False

        corpus = self.read_corpus(corpus_file)
        if corpus is None:
            return False
        
        self.word_probs, self.bigram_counts, self.first_word_counts, self.vocab = self.build_language_model(corpus)
        self.model_loaded = True
        return True
    
    def get_edits(self, word):
        """Generate edit distance 1 candidates"""
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        
        return set(deletes + transposes + replaces + inserts)
    
    def get_edits2(self, word):
        """Generate edit distance 2 candidates"""
        return set(e2 for e1 in self.get_edits(word) for e2 in self.get_edits(e1))
    
    def get_word_suggestions(self, word, max_suggestions=5):
        """Get ranked suggestions for a misspelled word"""
        if not self.model_loaded or word in self.vocab:
            return []
        
        # Get candidates
        candidates = self.get_edits(word) & self.vocab
        if not candidates:
            candidates = self.get_edits2(word) & self.vocab
        if not candidates:
            return []
        
        suggestions = []
        
        for candidate in candidates:
            log_word_prob = math.log(self.word_probs.get(candidate, 1e-10))
            suggestions.append((candidate, log_word_prob))
        
        # Sort by probability and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in suggestions[:max_suggestions]]
    
    def is_misspelled(self, word):
        """Check if a word is misspelled"""
        if not self.model_loaded:
            return False
        return word.lower() not in self.vocab and word.isalpha()
    
    def check_text(self, text):
        """Check entire text and return misspelled words with suggestions"""
        if not self.model_loaded:
            return {}
        
        words = re.findall(r'\b\w+\b', text)
        misspelled_data = {}
        
        for word in words:
            word_lower = word.lower()
            if self.is_misspelled(word_lower):
                suggestions = self.get_word_suggestions(word_lower)
                if suggestions:
                    misspelled_data[word] = suggestions
        
        return misspelled_data

class ModernSpellChecker:
    # --- UI Constants: Dark & Funky Theme ---
    BG_COLOR = "#1e1e2e"          # Dark purple/blue
    CARD_COLOR = "#313244"        # Lighter grey/purple
    ACCENT_COLOR = "#cba6f7"       # Lavender
    ACCENT_HOVER_COLOR = "#e8a2ed"  # Lighter lavender for hover
    ACCENT_COLOR_2 = "#f5c2e7"     # Pink
    TEXT_COLOR = "#cdd6f4"        # Main text
    SUBTLE_TEXT_COLOR = "#a6adc8"  # Subtext
    ERROR_COLOR = "#f38ba8"       # Red
    SUCCESS_COLOR = "#a6e3a1"     # Green
    WARNING_COLOR = "#fab387"     # Orange

    FONT_TITLE = ("Bahnschrift", 24, "bold")
    FONT_SUBTITLE = ("Bahnschrift", 11)
    FONT_BODY_BOLD = ("Bahnschrift", 11, "bold")
    FONT_BODY = ("Bahnschrift", 10)
    FONT_TEXT_AREA = ("Cascadia Code", 12)

    def __init__(self, root):
        self.root = root
        self.model = SpellCheckerModel()
        self.setup_ui()
        self.misspelled_data = {}
        self.misspelled_tags = []
        
        # Load model in background
        self.load_model_async()
    
    def setup_ui(self):
        """Setup the dark and funky UI"""
        self.root.title("QuickQuill")
        self.root.geometry("1100x750")
        self.root.configure(bg=self.BG_COLOR)
        
        # --- Configure Funky Styling ---
        style = ttk.Style()
        style.theme_use('clam')

        # General widget styling
        style.configure('.', background=self.BG_COLOR, foreground=self.TEXT_COLOR, font=self.FONT_BODY)
        style.configure('TFrame', background=self.BG_COLOR)
        style.configure('TLabel', background=self.BG_COLOR, foreground=self.TEXT_COLOR)

        # Title and Subtitle labels
        style.configure('Title.TLabel', font=self.FONT_TITLE, foreground=self.ACCENT_COLOR_2)
        style.configure('Subtitle.TLabel', font=self.FONT_SUBTITLE, foreground=self.SUBTLE_TEXT_COLOR)

        # Buttons
        style.configure('TButton', font=self.FONT_BODY_BOLD, background=self.CARD_COLOR, foreground=self.ACCENT_COLOR,
                        borderwidth=1, relief='solid', bordercolor=self.ACCENT_COLOR)
        style.map('TButton',
                  background=[('active', self.ACCENT_HOVER_COLOR), ('!disabled', self.CARD_COLOR)],
                  foreground=[('active', self.CARD_COLOR), ('!disabled', self.ACCENT_COLOR)])

        # Progress bar
        style.configure('TProgressbar', thickness=5, background=self.ACCENT_COLOR, troughcolor=self.CARD_COLOR)

        # Treeview (for suggestions)
        style.configure('Treeview', background=self.CARD_COLOR, fieldbackground=self.CARD_COLOR,
                        foreground=self.TEXT_COLOR, rowheight=25, font=self.FONT_BODY)
        style.configure('Treeview.Heading', background=self.BG_COLOR, font=self.FONT_BODY_BOLD,
                        foreground=self.ACCENT_COLOR)
        style.map('Treeview', background=[('selected', self.ACCENT_COLOR)], foreground=[('selected', self.BG_COLOR)])

        # Scrollbar
        style.configure('Vertical.TScrollbar', background=self.BG_COLOR, bordercolor=self.BG_COLOR,
                        troughcolor=self.BG_COLOR, arrowcolor=self.ACCENT_COLOR)
        style.map('Vertical.TScrollbar', background=[('active', self.CARD_COLOR)])
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        self.create_header(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        # Left panel - Text input
        left_panel = tk.Frame(content_frame, bg=self.ACCENT_COLOR_2, bd=1, relief='solid')
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        left_panel_inner = ttk.Frame(left_panel, style='Card.TFrame')
        left_panel_inner.pack(fill='both', expand=True)
        style.configure('Card.TFrame', background=self.CARD_COLOR)
        
        self.create_text_input_panel(left_panel_inner)
        
        # Right panel - Results
        right_panel = tk.Frame(content_frame, bg=self.ACCENT_COLOR, bd=1, relief='solid')
        right_panel.pack(side='right', fill='both', padx=(10, 0))
        right_panel_inner = ttk.Frame(right_panel, style='Card.TFrame')
        right_panel_inner.pack(fill='both', expand=True)

        self.create_results_panel(right_panel_inner)
    
    def create_header(self, parent):
        """Create the header section"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="<QuickQuill/>", style='Title.TLabel')
        title_label.pack(anchor='w')
        
        subtitle_label = ttk.Label(header_frame, text="Spell checking for the modern wordsmith", style='Subtitle.TLabel')
        subtitle_label.pack(anchor='w', pady=(0, 5))
    
    def create_status_bar(self, parent):
        """Create status bar"""
        self.status_frame = ttk.Frame(parent)
        self.status_frame.pack(fill='x')
        
        self.status_label = ttk.Label(self.status_frame, text="Initializing system...", font=self.FONT_BODY, foreground=self.WARNING_COLOR)
        self.status_label.pack(side='left')
        
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress.pack(side='right', fill='x', expand=True, padx=10)
        self.progress.start(10)
    
    def create_text_input_panel(self, parent):
        """Create the text input panel"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', padx=15, pady=(15, 10))
        
        input_label = ttk.Label(header_frame, text="📝 Input Matrix", font=self.FONT_BODY_BOLD, foreground=self.TEXT_COLOR)
        input_label.pack(side='left')
        
        buttons_frame = ttk.Frame(header_frame)
        buttons_frame.pack(side='right')
        
        clear_btn = ttk.Button(buttons_frame, text="🗑️ Purge", command=self.clear_text, width=8)
        clear_btn.pack(side='right', padx=(5, 0))
        
        check_btn = ttk.Button(buttons_frame, text="✓ Execute", command=self.check_spelling, width=12)
        check_btn.pack(side='right')
        
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        self.text_area = tk.Text(text_frame, wrap='word', font=self.FONT_TEXT_AREA,
                                bg=self.CARD_COLOR, fg=self.TEXT_COLOR, relief='flat',
                                selectbackground=self.ACCENT_COLOR, selectforeground=self.BG_COLOR,
                                insertbackground=self.ACCENT_COLOR_2, # Blinking cursor color
                                padx=10, pady=10, height=15, bd=0)
        
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.text_area.yview, style='Vertical.TScrollbar')
        self.text_area.configure(yscrollcommand=scrollbar.set)
        
        self.text_area.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.text_area.bind('<KeyRelease>', self.on_text_change)
        self.text_area.bind('<Button-1>', self.on_click)
        
        self.create_examples_section(parent)
        self.setup_text_tags()
    
    def create_examples_section(self, parent):
        """Create examples section"""
        examples_frame = ttk.Frame(parent)
        examples_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        examples_label = ttk.Label(examples_frame, text="📋 Quick Payloads:", font=self.FONT_BODY_BOLD)
        examples_label.pack(anchor='w', pady=(0, 5))
        
        examples = [
            "I havve a gret idee for tomorow",
            "The quik brown fox jumps over the lasy dog",
            "Ths is a vrey importnt mesage",
            "I wnt to the stor yestrday"
        ]
        
        examples_grid = ttk.Frame(examples_frame)
        examples_grid.pack(fill='x')
        
        for i, example in enumerate(examples):
            row = i // 2
            col = i % 2
            
            btn = ttk.Button(examples_grid, text=f"{example[:25]}...", command=lambda ex=example: self.load_example(ex), width=30)
            btn.grid(row=row, column=col, padx=2, pady=2, sticky='ew')
        
        examples_grid.columnconfigure(0, weight=1)
        examples_grid.columnconfigure(1, weight=1)
    
    def create_results_panel(self, parent):
        """Create the results panel"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', padx=15, pady=(15, 10))
        
        results_label = ttk.Label(header_frame, text="🔍 Analysis Output", font=self.FONT_BODY_BOLD)
        results_label.pack(side='left')
        
        stats_frame = ttk.Frame(parent)
        stats_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.create_stats_widgets(stats_frame)
        
        suggestions_frame = ttk.Frame(parent)
        suggestions_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        suggestions_label = ttk.Label(suggestions_frame, text="💡 Correction Protocols", font=self.FONT_BODY_BOLD)
        suggestions_label.pack(anchor='w', pady=(0, 5))
        
        self.suggestions_frame_inner = ttk.Frame(suggestions_frame)
        self.suggestions_frame_inner.pack(fill='both', expand=True)
        
        self.suggestions_tree = ttk.Treeview(self.suggestions_frame_inner, columns=('suggestions',), show='tree headings')
        self.suggestions_tree.heading('#0', text='Anomaly', anchor='w')
        self.suggestions_tree.heading('suggestions', text='Suggestions', anchor='w')
        self.suggestions_tree.column('#0', width=120, minwidth=100, stretch=tk.NO)
        self.suggestions_tree.column('suggestions', width=200, minwidth=150)
        
        suggestions_scrollbar = ttk.Scrollbar(self.suggestions_frame_inner, orient='vertical', command=self.suggestions_tree.yview, style='Vertical.TScrollbar')
        self.suggestions_tree.configure(yscrollcommand=suggestions_scrollbar.set)
        
        self.suggestions_tree.pack(side='left', fill='both', expand=True)
        suggestions_scrollbar.pack(side='right', fill='y')
        
        self.suggestions_tree.bind('<Double-1>', self.on_suggestion_double_click)
    
    def create_stats_widgets(self, parent):
        """Create statistics widgets"""
        stats_container = ttk.Frame(parent)
        stats_container.pack(fill='x')
        
        self.total_words_var = tk.StringVar(value="0")
        self.misspelled_var = tk.StringVar(value="0")
        self.accuracy_var = tk.StringVar(value="100%")
        
        self.create_stat_card(stats_container, "📊 Total Words", self.total_words_var, 0)
        self.create_stat_card(stats_container, "❌ Wrong Words", self.misspelled_var, 1)
        self.create_stat_card(stats_container, "✅ Accuracy", self.accuracy_var, 2)
    
    def create_stat_card(self, parent, title, textvariable, column):
        """Create individual stat card with a border"""
        card_border = tk.Frame(parent, bg=self.ACCENT_COLOR, bd=0)
        card_border.grid(row=0, column=column, padx=2, pady=2, sticky='ew')
        
        card = ttk.Frame(card_border, style='Card.TFrame')
        card.pack(padx=1, pady=1)

        title_label = ttk.Label(card, text=title, font=self.FONT_BODY, foreground=self.SUBTLE_TEXT_COLOR)
        title_label.pack(pady=(8, 2))
        
        value_label = ttk.Label(card, textvariable=textvariable, font=("Digital-7 Mono", 18, "bold"), foreground=self.TEXT_COLOR)
        value_label.pack(pady=(0, 8))
        
        parent.columnconfigure(column, weight=1)
    
    def setup_text_tags(self):
        """Setup text highlighting tags for the dark theme"""
        self.text_area.tag_configure('misspelled', 
                                    foreground=self.ERROR_COLOR,
                                    underline=True)
    
    def load_model_async(self):
        """Load model in background thread"""
        def load():
            success = self.model.load_model()
            self.root.after(0, lambda: self.on_model_loaded(success))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def on_model_loaded(self, success):
        """Called when model loading completes"""
        self.progress.stop()
        self.progress.pack_forget()
        
        if success:
            self.status_label.configure(text="✅ System online. Ready for input.", foreground=self.SUCCESS_COLOR)
        else:
            self.status_label.configure(text="❌ Critical Error: Model 'big.txt' not found.", foreground=self.ERROR_COLOR)
            messagebox.showwarning(
                "Model Loading Error",
                "Could not load spell checker model.\n\n"
                "Please download 'big.txt' from:\n"
                "http://norvig.com/big.txt\n\n"
                "And place it in the same directory as this script."
            )
    
    def load_example(self, example):
        """Load example text"""
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(1.0, example)
        self.check_spelling()
    
    def clear_text(self):
        """Clear all text and highlighting"""
        self.text_area.delete(1.0, tk.END)
        self.clear_results()
    
    def clear_results(self):
        """Clear all results and highlighting"""
        for tag in self.misspelled_tags:
            self.text_area.tag_delete(tag)
        self.misspelled_tags.clear()
        
        for item in self.suggestions_tree.get_children():
            self.suggestions_tree.delete(item)
        
        self.total_words_var.set("0")
        self.misspelled_var.set("0")
        self.accuracy_var.set("100%")
        
        self.misspelled_data.clear()
    
    def on_text_change(self, event=None):
        """Handle text change events"""
        if hasattr(self, '_check_timer'):
            self.root.after_cancel(self._check_timer)
        self._check_timer = self.root.after(750, self.check_spelling)
    
    def on_click(self, event):
        """Handle click on text area"""
        index = self.text_area.index(f"@{event.x},{event.y}")
        tags = self.text_area.tag_names(index)
        for tag in tags:
            if tag.startswith('misspelled_'):
                word = tag.split('_', 1)[1]
                self.show_word_context_menu(event, word, index)
                break
    
    def show_word_context_menu(self, event, word, index):
        """Show context menu for misspelled word"""
        if word not in self.misspelled_data:
            return
        
        context_menu = tk.Menu(self.root, tearoff=0, bg=self.CARD_COLOR, fg=self.TEXT_COLOR,
                               activebackground=self.ACCENT_COLOR, activeforeground=self.BG_COLOR,
                               relief='solid', bd=1, font=self.FONT_BODY)
        suggestions = self.misspelled_data[word][:5]
        
        for suggestion in suggestions:
            context_menu.add_command(label=f"✓ {suggestion}", command=lambda s=suggestion, w=word: self.replace_word(w, s))
        
        if suggestions:
            context_menu.add_separator(background=self.ACCENT_COLOR)
        
        context_menu.add_command(label="╳ Ignore", command=lambda: None)
        
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    def check_spelling(self):
        """Check spelling of current text"""
        if not self.model.model_loaded:
            return
        
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            self.clear_results()
            return
        
        self.clear_results()
        self.misspelled_data = self.model.check_text(text)
        
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)
        misspelled_count = len(self.misspelled_data)
        accuracy = round((1 - misspelled_count / max(1, total_words)) * 100, 1) if total_words > 0 else 100
        
        self.total_words_var.set(str(total_words))
        self.misspelled_var.set(str(misspelled_count))
        self.accuracy_var.set(f"{accuracy}%")
        
        self.highlight_misspelled_words(text)
        self.update_suggestions_tree()
        
        if misspelled_count == 0:
            self.status_label.configure(text="✅ Integrity at 100%. No anomalies detected.", foreground=self.SUCCESS_COLOR)
        else:
            self.status_label.configure(text=f"🔍 Found {misspelled_count} anomaly(s). Awaiting correction.", foreground=self.WARNING_COLOR)
    
    def highlight_misspelled_words(self, text):
        """Highlight misspelled words in the text"""
        words = re.finditer(r'\b\w+\b', text)
        
        for match in words:
            word = match.group()
            if word in self.misspelled_data:
                start_idx = f"1.0+{match.start()}c"
                end_idx = f"1.0+{match.end()}c"
                
                tag_name = f'misspelled_{word}'
                self.misspelled_tags.append(tag_name)
                
                self.text_area.tag_add(tag_name, start_idx, end_idx)
                self.text_area.tag_configure(tag_name, foreground=self.ERROR_COLOR, underline=True)
    
    def update_suggestions_tree(self):
        """Update the suggestions tree"""
        for word, suggestions in self.misspelled_data.items():
            suggestions_text = " • ".join(suggestions[:3])
            if len(suggestions) > 3:
                suggestions_text += f" (+{len(suggestions)-3})"
            
            item = self.suggestions_tree.insert('', 'end', text=word, values=(suggestions_text,))
            
            for suggestion in suggestions[:5]:
                self.suggestions_tree.insert(item, 'end', text=f"  ↳ {suggestion}", values=('',))
    
    def on_suggestion_double_click(self, event):
        """Handle double-click on suggestion"""
        item = self.suggestions_tree.selection()[0]
        text = self.suggestions_tree.item(item, 'text')
        
        if text.startswith('  ↳ '):
            suggestion = text[4:]
            parent_item = self.suggestions_tree.parent(item)
            original_word = self.suggestions_tree.item(parent_item, 'text')
            self.replace_word(original_word, suggestion)
        elif not self.suggestions_tree.parent(item):
            if text in self.misspelled_data and self.misspelled_data[text]:
                self.replace_word(text, self.misspelled_data[text][0])
    
    def replace_word(self, original_word, replacement):
        """Replace misspelled word with suggestion"""
        text = self.text_area.get(1.0, tk.END)
        pattern = r'\b' + re.escape(original_word) + r'\b'
        
        # Preserve case: if original was capitalized, capitalize replacement
        if original_word.istitle():
            replacement = replacement.title()
        elif original_word.isupper():
            replacement = replacement.upper()

        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(1.0, new_text)
        
        # Re-check spelling automatically after replacement
        self.root.after(100, self.check_spelling)

def main():
    """Main function to run the application"""
    root = tk.Tk()
    root.resizable(True, True)
    root.minsize(900, 650)
    
    root.update_idletasks()
    width = 1100
    height = 750
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    app = ModernSpellChecker(root)
    root.mainloop()

if __name__ == "__main__":
    main()