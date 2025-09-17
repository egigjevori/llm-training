#!/usr/bin/env python3
"""
Beautiful Chat CLI Application
Generic chat interface with multiple response providers (Qdrant, Model, Hybrid RAG)
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
import time
import random

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich.spinner import Spinner
from rich.live import Live
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich import box

from providers import get_provider, ResponseProvider

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise in CLI
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

# Enhanced ASCII Art Banner with Emojis
BANNER = """
🌟 ██████╗██╗  ██╗ █████╗ ████████╗     ██████╗██╗     ██╗ 🌟
✨ ██╔════╝██║  ██║██╔══██╗╚══██╔══╝    ██╔════╝██║     ██║ ✨
💫 ██║     ███████║███████║   ██║       ██║     ██║     ██║ 💫
🚀 ██║     ██╔══██║██╔══██║   ██║       ██║     ██║     ██║ 🚀
⭐ ╚██████╗██║  ██║██║  ██║   ██║       ╚██████╗███████╗██║ ⭐
🎯  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╝╝        ╚═════╝╚══════╝╚═╝  🎯
"""

# Emoji collections for various purposes
THINKING_EMOJIS = ['🤔', '💭', '🧠', '⚡', '✨', '🎯', '🔍', '🤖']
SUCCESS_EMOJIS = ['✅', '🎉', '🌟', '💫', '🚀', '⭐', '✨', '🎊']
ERROR_EMOJIS = ['❌', '🚨', '⚠️', '💥', '🔥', '😵', '🙈', '💀']
LOADING_EMOJIS = ['⏳', '⌛', '🔄', '🔀', '⚡', '💫', '🌀', '🎯']

# Animation frames for various effects
SPINNER_FRAMES = {
    'dots': ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'],
    'aesthetic': ['✨', '💫', '⭐', '🌟', '💎', '🔮', '🌈', '🦄'],
    'tech': ['🔄', '⚡', '🔁', '💻', '🤖', '🎯', '🚀', '⚙️'],
    'stars': ['⋆', '✦', '✧', '⋄', '◆', '◇', '❖', '✪']
}


class ChatSession:
    """Manages a chat session with a response provider."""
    
    def __init__(self, provider: ResponseProvider, mode: str):
        self.provider = provider
        self.mode = mode
        self.conversation_history = []
        
    def display_banner(self):
        """Display the enhanced application banner with animations."""
        # Animated startup sequence
        self._animate_startup()
        
        # Enhanced banner with gradient colors and emojis
        banner_colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
        banner_text = Text(BANNER)
        
        # Apply gradient effect to banner
        for i, line in enumerate(BANNER.split('\n')):
            color = banner_colors[i % len(banner_colors)]
            banner_text = Text(BANNER, style=f"bold {color}")
        
        console.print(Panel(
            Align.center(banner_text),
            border_style="bright_cyan",
            padding=(1, 2),
            box=box.DOUBLE_EDGE
        ))
        
        # Enhanced mode information with emojis
        mode_info = Table.grid(padding=1)
        mode_info.add_column(style="bold bright_blue", justify="right")
        mode_info.add_column()
        
        # Add emojis based on mode
        mode_emoji = "🤖" if self.mode == "model" else "🔍" if self.mode == "qdrant" else "🔄"
        provider_emoji = "⚡" if "model" in self.provider.get_name().lower() else "🎯"
        
        mode_info.add_row(f"{mode_emoji} Mode:", f"[bold bright_green]{self.mode.upper()}[/bold bright_green]")
        mode_info.add_row(f"{provider_emoji} Provider:", f"[bright_yellow]{self.provider.get_name()}[/bright_yellow]")
        mode_info.add_row("🎮 Commands:", "[dim bright_white]/help, /mode, /history, /clear, /quit[/dim bright_white]")
        
        console.print(Panel(
            mode_info,
            title="[bold bright_magenta]✨ Chat Session Info ✨[/bold bright_magenta]",
            border_style="bright_blue",
            padding=(0, 1),
            box=box.ROUNDED
        ))
        
        # Add decorative separator
        console.print(Rule("[bright_cyan]🌟 Ready to Chat! 🌟[/bright_cyan]", style="bright_cyan"))
    
    def _animate_startup(self):
        """Display animated startup sequence."""
        startup_messages = [
            "🔥 Initializing Chat CLI...",
            "⚡ Loading AI systems...",
            "🌟 Preparing interface...",
            "🚀 Ready to launch!"
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            for message in startup_messages:
                task = progress.add_task(message, total=1)
                time.sleep(0.3)
                progress.update(task, advance=1)
                time.sleep(0.2)
        
    def display_typing_animation(self, duration: float = 2.0, animation_type: str = "auto"):
        """Display enhanced typing animation with multiple styles."""
        # Choose animation style based on provider type or random
        if animation_type == "auto":
            if "model" in self.provider.get_name().lower():
                animation_type = "tech"
            elif "qdrant" in self.provider.get_name().lower():
                animation_type = "aesthetic"
            else:
                animation_type = random.choice(['dots', 'aesthetic', 'tech', 'stars'])
        
        # Choose thinking messages and emojis
        thinking_messages = [
            f"{random.choice(THINKING_EMOJIS)} Analyzing your question...",
            f"{random.choice(THINKING_EMOJIS)} Processing information...",
            f"{random.choice(THINKING_EMOJIS)} Generating response...",
            f"{random.choice(THINKING_EMOJIS)} Almost ready...",
        ]
        
        # Enhanced spinner options
        spinner_configs = {
            'dots': ("dots", "bright_cyan"),
            'aesthetic': ("dots2", "bright_magenta"), 
            'tech': ("line", "bright_green"),
            'stars': ("star", "bright_yellow")
        }
        
        spinner_name, spinner_color = spinner_configs.get(animation_type, ("dots", "cyan"))
        
        with Live(console=console, refresh_per_second=4) as live:
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration:
                # Cycle through thinking messages
                message_index = int((frame_count // 8) % len(thinking_messages))
                current_message = thinking_messages[message_index]
                
                # Create animated spinner with changing messages
                spinner = Spinner(spinner_name, text=f"[{spinner_color}]{current_message}[/{spinner_color}]", style=spinner_color)
                
                # Add progress indicator dots
                dots_count = (frame_count % 12) // 3
                progress_dots = "." * (dots_count + 1) + " " * (3 - dots_count)
                
                # Create spinner display with text
                spinner_display = Spinner(spinner_name, text=f"[{spinner_color}]{current_message}[/{spinner_color}]", style=spinner_color)
                
                # Create content with progress dots
                full_content = Text()
                full_content.append(current_message, style=spinner_color)
                full_content.append(f"\n{progress_dots}", style="dim")
                
                live.update(Panel(
                    Align.center(full_content),
                    border_style=spinner_color,
                    padding=(1, 2),
                    box=box.ROUNDED,
                    title=f"[{spinner_color}]{random.choice(LOADING_EMOJIS)}[/{spinner_color}]"
                ))
                
                time.sleep(0.25)
                frame_count += 1
    
    def typewriter_effect(self, text: str, delay: float = 0.02, style: str = "bright_white"):
        """Display text with typewriter effect."""
        if len(text) > 200:  # Skip typewriter for very long text
            return Text(text, style=style)
        
        displayed_text = Text(style=style)
        
        with Live(displayed_text, refresh_per_second=20, console=console, transient=True) as live:
            for char in text:
                displayed_text.append(char)
                live.update(displayed_text)
                time.sleep(delay)
        
        return Text(text, style=style)
    
    def show_transition_effect(self, message: str = "Processing..."):
        """Show a brief transition effect."""
        transition_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        with Live(console=console, refresh_per_second=10, transient=True) as live:
            for i in range(8):  # Brief animation
                char = transition_chars[i % len(transition_chars)]
                live.update(Text(f"{char} {message}", style="bright_cyan"))
                time.sleep(0.1)
    
    def format_response(self, response_data: Dict[str, Any]) -> Panel:
        """Format the response in a beautiful panel with emojis and enhanced styling."""
        response = response_data.get('response', '')
        source = response_data.get('source', 'unknown')
        metadata = response_data.get('metadata', {})
        
        # Create response content
        if response.strip():
            try:
                # Try to render as markdown if it looks like it has markdown formatting
                if any(marker in response for marker in ['**', '#', '*', '`', '\n-', '\n*']):
                    content = Markdown(response)
                else:
                    content = Text(response, style="bright_white")
            except:
                content = Text(response, style="bright_white")
        else:
            content = Text(f"{random.choice(ERROR_EMOJIS)} No response generated.", style="dim red")
        
        # Enhanced source-specific styling and emojis
        source_config = {
            'qdrant': {
                'emoji': '🔍',
                'color': 'bright_blue',
                'border': 'bright_blue',
                'style': 'bold bright_blue'
            },
            'model': {
                'emoji': '🤖',
                'color': 'bright_green', 
                'border': 'bright_green',
                'style': 'bold bright_green'
            },
            'hybrid': {
                'emoji': '⚡',
                'color': 'bright_magenta',
                'border': 'bright_magenta', 
                'style': 'bold bright_magenta'
            }
        }
        
        config = source_config.get(source, {
            'emoji': '❓',
            'color': 'white',
            'border': 'white',
            'style': 'bold white'
        })
        
        # Create enhanced metadata footer with emojis
        footer_parts = [f"[dim]{config['emoji']} Source: [{config['color']}]{source.upper()}[/{config['color']}][/dim]"]
        
        if source == 'qdrant' and 'results_found' in metadata:
            results_emoji = '📊' if metadata['results_found'] > 0 else '❌'
            footer_parts.append(f"[dim]{results_emoji} Results: {metadata['results_found']}[/dim]")
            if 'top_score' in metadata:
                score_emoji = '⭐' if metadata['top_score'] > 0.8 else '📈' if metadata['top_score'] > 0.5 else '📉'
                footer_parts.append(f"[dim]{score_emoji} Score: {metadata['top_score']:.3f}[/dim]")
        elif source == 'model' and 'temperature' in metadata:
            temp_emoji = '🔥' if metadata['temperature'] > 0.7 else '❄️' if metadata['temperature'] < 0.3 else '🌡️'
            footer_parts.append(f"[dim]{temp_emoji} Temp: {metadata['temperature']}[/dim]")
        
        footer = " • ".join(footer_parts)
        
        # Enhanced title with emoji and styling
        title = f"[{config['style']}]{config['emoji']} {self.provider.get_name()} {random.choice(SUCCESS_EMOJIS)}[/{config['style']}]"
        
        return Panel(
            content,
            title=title,
            subtitle=footer,
            border_style=config['border'],
            padding=(1, 2),
            box=box.ROUNDED
        )
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True to continue, False to quit."""
        command = command.lower().strip()
        
        if command in ['/quit', '/exit', '/q']:
            farewell_messages = [
                f"{random.choice(SUCCESS_EMOJIS)} Thanks for chatting! See you next time!",
                f"🚀 Thanks for using Chat CLI! Have a great day!",
                f"✨ Chat session completed! Until next time!",
                f"🌟 Goodbye! Keep exploring and asking questions!"
            ]
            console.print(f"\n[bold bright_cyan]{random.choice(farewell_messages)}[/bold bright_cyan]\n")
            return False
            
        elif command == '/help':
            help_table = Table.grid(padding=1)
            help_table.add_column(style="bold bright_cyan", justify="right")
            help_table.add_column()
            
            commands = [
                ("❓ /help", "Show this help message"),
                ("⚙️ /mode", "Display current mode information"),  
                ("📜 /history", "Show conversation history"),
                ("🗑️ /clear", "Clear conversation history"),
                ("👋 /quit", "Exit the chat application")
            ]
            
            for cmd, desc in commands:
                help_table.add_row(cmd, desc)
            
            console.print(Panel(
                help_table,
                title="[bold bright_magenta]🎮 Available Commands 🎮[/bold bright_magenta]",
                border_style="bright_cyan",
                box=box.ROUNDED
            ))
            
        elif command == '/mode':
            mode_table = Table.grid(padding=1)
            mode_table.add_column(style="bold bright_blue", justify="right")
            mode_table.add_column()
            
            # Enhanced mode display with emojis
            mode_emoji = "🤖" if self.mode == "model" else "🔍" if self.mode == "qdrant" else "⚡"
            provider_emoji = "🧠" if "model" in self.provider.get_name().lower() else "📚"
            
            mode_table.add_row(f"{mode_emoji} Mode:", f"[bold bright_green]{self.mode.upper()}[/bold bright_green]")
            mode_table.add_row(f"{provider_emoji} Provider:", f"[bright_yellow]{self.provider.get_name()}[/bright_yellow]")
            
            if hasattr(self.provider, 'collection_name'):
                mode_table.add_row("📊 Collection:", f"[bright_cyan]{self.provider.collection_name}[/bright_cyan]")
            
            console.print(Panel(
                mode_table,
                title="[bold bright_blue]⚙️ Current Mode Info ⚙️[/bold bright_blue]",
                border_style="bright_blue",
                box=box.ROUNDED
            ))
            
        elif command == '/history':
            if not self.conversation_history:
                console.print(f"[dim bright_yellow]📜 No conversation history yet. Start chatting! {random.choice(SUCCESS_EMOJIS)}[/dim bright_yellow]")
            else:
                history_content = []
                for i, item in enumerate(self.conversation_history[-5:], 1):
                    history_content.append(f"[bold bright_blue]💬 Q{i}:[/bold bright_blue] {item['query']}")
                    response_preview = item['response'][:100] + ('...' if len(item['response']) > 100 else '')
                    history_content.append(f"[bold bright_green]🤖 A{i}:[/bold bright_green] {response_preview}")
                    history_content.append("")  # Add spacing
                
                history_panel = Panel(
                    "\n".join(history_content),
                    title="[bold bright_yellow]📜 Recent Conversation History 📜[/bold bright_yellow]",
                    border_style="bright_yellow",
                    box=box.ROUNDED
                )
                console.print(history_panel)
                
        elif command == '/clear':
            self.conversation_history.clear()
            console.clear()
            self.display_banner()
            clear_messages = [
                f"{random.choice(SUCCESS_EMOJIS)} Conversation history cleared! Fresh start!",
                f"🧹 All clean! Ready for new conversations!",
                f"✨ History wiped! Let's start over!",
                f"🗑️ Cleared! Time for new adventures!"
            ]
            console.print(f"[bright_green]{random.choice(clear_messages)}[/bright_green]")
            
        else:
            console.print(f"[red]{random.choice(ERROR_EMOJIS)} Unknown command: {command}[/red]")
            console.print("[dim bright_white]💡 Type /help for available commands[/dim bright_white]")
        
        return True
    
    def run(self):
        """Run the main chat loop."""
        console.clear()
        self.display_banner()
        
        welcome_messages = [
            "🚀 Welcome to Chat CLI! Let's explore together!",
            "✨ Ready to chat! Ask me anything!", 
            "🎉 Chat CLI is ready! What's on your mind?",
            "💫 Hello! Let's have an amazing conversation!"
        ]
        console.print(f"\n[bold bright_green]{random.choice(welcome_messages)}[/bold bright_green]")
        console.print("[dim bright_white]💡 Type your questions or use commands like /help, /quit[/dim bright_white]\n")
        
        try:
            while True:
                try:
                    # Dynamic input prompts with emojis
                    input_prompts = ["💬", "🗨️", "💭", "📝", "✍️", "🎯"]
                    current_prompt = random.choice(input_prompts)
                    
                    # Get user input with dynamic prompt
                    user_input = input(f"{current_prompt} ").strip()
                    
                    if not user_input:
                        # Encourage user with random messages
                        encouragement = [
                            "🤔 Got a question? I'm here to help!",
                            "✨ What would you like to know?", 
                            "🚀 Ready when you are!",
                            "💫 Type something to get started!"
                        ]
                        console.print(f"[dim bright_cyan]{random.choice(encouragement)}[/dim bright_cyan]")
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        if not self.handle_command(user_input):
                            break
                        continue
                    
                    # Process regular query
                    console.print()  # Add spacing
                    
                    # Show typing animation
                    self.display_typing_animation(1.5)
                    
                    # Show transition effect
                    self.show_transition_effect(f"Generating response {random.choice(LOADING_EMOJIS)}")
                    
                    # Get response from provider
                    try:
                        response_data = self.provider.get_response(user_input)
                        
                        # Add a brief pause for dramatic effect
                        time.sleep(0.3)
                        
                        # Add spacing before response
                        console.print()
                        
                        # Display response with enhanced formatting
                        response_panel = self.format_response(response_data)
                        console.print(response_panel)
                        
                        # Store in history
                        self.conversation_history.append({
                            'query': user_input,
                            'response': response_data.get('response', ''),
                            'source': response_data.get('source', 'unknown'),
                            'metadata': response_data.get('metadata', {})
                        })
                        
                    except Exception as e:
                        error_messages = [
                            f"{random.choice(ERROR_EMOJIS)} Oops! Something went wrong while processing your request.",
                            f"{random.choice(ERROR_EMOJIS)} Sorry, I encountered an issue. Let me help you try again!",
                            f"{random.choice(ERROR_EMOJIS)} Hmm, that didn't work as expected. Don't worry, we'll figure it out!",
                            f"{random.choice(ERROR_EMOJIS)} Technical hiccup detected! Let's give it another shot."
                        ]
                        
                        console.print(Panel(
                            Text.assemble(
                                (f"{random.choice(error_messages)}\n\n", "bright_red"),
                                ("Technical details: ", "dim"),
                                (str(e), "dim red")
                            ),
                            title=f"[bold bright_red]{random.choice(ERROR_EMOJIS)} Error {random.choice(ERROR_EMOJIS)}[/bold bright_red]",
                            border_style="bright_red",
                            box=box.ROUNDED
                        ))
                    
                    console.print()  # Add spacing
                    
                except KeyboardInterrupt:
                    interrupt_messages = [
                        "⚠️ Interrupted! Use /quit to exit gracefully",
                        "🛑 Hold on! Type /quit to exit properly", 
                        "⏸️ Paused! Use /quit for a clean exit",
                        "🚨 Ctrl+C detected! Try /quit instead"
                    ]
                    console.print(f"\n[bright_yellow]{random.choice(interrupt_messages)}[/bright_yellow]")
                    continue
                except EOFError:
                    console.print(f"\n[dim bright_cyan]👋 Session ended. Until next time! {random.choice(SUCCESS_EMOJIS)}[/dim bright_cyan]")
                    break
                    
        except Exception as e:
            console.print(Panel(
                Text.assemble(
                    (f"{random.choice(ERROR_EMOJIS)} An unexpected error occurred!\n\n", "bright_red"),
                    ("Error details: ", "dim"),
                    (str(e), "dim red")
                ),
                title=f"[bold bright_red]💥 Critical Error 💥[/bold bright_red]",
                border_style="bright_red",
                box=box.ROUNDED
            ))
        
        # Enhanced session end message
        end_messages = [
            f"{random.choice(SUCCESS_EMOJIS)} Chat session completed! Thanks for using Chat CLI!",
            f"👋 Session ended gracefully. Come back soon!",
            f"✨ Chat CLI session finished. See you next time!",
            f"🌟 Thanks for chatting! Until we meet again!"
        ]
        console.print(f"\n[dim bright_green]{random.choice(end_messages)}[/dim bright_green]")


def main():
    """
    Beautiful Chat CLI Application
    
    A generic chat interface supporting multiple response providers:
    - Qdrant: Vector search and retrieval
    - Model: Fine-tuned model inference
    """
    
    # Configuration - modify these variables as needed
    # mode = 'model'  # Options: 'qdrant' or 'model'
    mode = 'qdrant'

    # Qdrant configuration (for mode='qdrant')
    # collection = 'corporate_data'
    # top_k = 3
    
    # Model configuration (for mode='model') 
    model_path = '../models/finetuned_model'
    base_model = 'Qwen/Qwen2.5-1.5B-Instruct'
    # temperature = 0.7
    # max_tokens = 200
    
    try:
        # Prepare provider parameters
        provider_kwargs = {}
        
        if mode == 'qdrant':
            provider_kwargs.update({
                'collection_name': 'corporate_data',  # collection
                'top_k': 3  # top_k
            })
            
        elif mode == 'model':
            provider_kwargs.update({
                'model_path': model_path,
                'base_model': base_model
            })
        
        # Create provider
        provider = get_provider(mode, **provider_kwargs)
        
        # Create and run chat session
        chat = ChatSession(provider, mode)
        chat.run()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        print(f"\nError starting chat application: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()