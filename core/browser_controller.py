#!/usr/bin/env python3
"""
Browser Controller module for DEFIMIND

This module enables the agent to control a web browser, automate web tasks,
and interact with DeFi platforms and web interfaces.
"""

import os
import sys
import time
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import re
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/browser_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BrowserController")

class BrowserController:
    """
    BrowserController enables the agent to control a web browser, execute tasks,
    and collect data from the web.
    """
    
    def __init__(self, headless: bool = False, browser_type: str = "chrome", 
                 user_data_dir: str = None, timeout: int = 30):
        """
        Initialize the BrowserController with specified settings.
        
        Args:
            headless: Whether to run the browser in headless mode (no UI)
            browser_type: The type of browser to use (chrome, firefox, safari)
            user_data_dir: Path to a custom user data directory to preserve browser state
            timeout: Default timeout in seconds for browser operations
        """
        self.headless = headless
        self.browser_type = browser_type.lower()
        self.user_data_dir = user_data_dir
        self.timeout = timeout
        
        # Will store the main browser object when initialized
        self.browser = None
        
        # Will hold current page/tab objects
        self.active_page = None
        self.pages = []
        
        # Track browser session stats
        self.session_stats = {
            "pages_visited": 0,
            "errors": 0,
            "start_time": None,
            "last_action_time": None,
            "screenshots_taken": 0,
            "data_scraped": 0,
            "forms_submitted": 0,
            "login_attempts": 0
        }
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Initialize dependencies
        self._initialize_dependencies()
        
        logger.info(f"BrowserController initialized with {browser_type} browser")
    
    def _initialize_dependencies(self):
        """
        Check for and install necessary dependencies for browser automation
        """
        try:
            # Try importing the required modules
            import playwright
            logger.info("Playwright module already installed")
        except ImportError:
            # Install playwright if not available
            logger.info("Installing playwright module")
            import subprocess
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], 
                               check=True, capture_output=True)
                subprocess.run([sys.executable, "-m", "playwright", "install"], 
                               check=True, capture_output=True)
                logger.info("Playwright installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install playwright: {e.stderr.decode()}")
                raise RuntimeError("Failed to install browser automation dependencies")
    
    async def start_browser(self) -> bool:
        """
        Start the browser with configured settings
        
        Returns:
            bool: True if browser was successfully started
        """
        from playwright.async_api import async_playwright
        
        logger.info(f"Starting {self.browser_type} browser (headless={self.headless})")
        
        try:
            # Initialize Playwright
            self.playwright = await async_playwright().start()
            
            # Select browser type
            if self.browser_type == "firefox":
                browser_class = self.playwright.firefox
            elif self.browser_type == "webkit" or self.browser_type == "safari":
                browser_class = self.playwright.webkit
            else:
                # Default to Chromium
                browser_class = self.playwright.chromium
            
            # Set browser launch options
            launch_options = {
                "headless": self.headless,
                "timeout": self.timeout * 1000  # Convert to ms
            }
            
            if self.user_data_dir:
                launch_options["user_data_dir"] = self.user_data_dir
            
            # Launch browser
            self.browser = await browser_class.launch(**launch_options)
            
            # Create initial page
            self.active_page = await self.browser.new_page()
            self.pages = [self.active_page]
            
            # Set session start time
            self.session_stats["start_time"] = datetime.now().isoformat()
            self.session_stats["last_action_time"] = self.session_stats["start_time"]
            
            logger.info("Browser started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start browser: {str(e)}")
            logger.error(traceback.format_exc())
            self.session_stats["errors"] += 1
            return False
    
    async def close_browser(self) -> bool:
        """
        Close the browser and clean up resources
        
        Returns:
            bool: True if browser was successfully closed
        """
        if not self.browser:
            logger.warning("No browser instance to close")
            return False
        
        try:
            await self.browser.close()
            await self.playwright.stop()
            
            logger.info("Browser closed successfully")
            return True
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
            self.session_stats["errors"] += 1
            return False
    
    async def navigate(self, url: str, wait_until: str = "load") -> bool:
        """
        Navigate to a URL in the active page
        
        Args:
            url: The URL to navigate to
            wait_until: When to consider navigation complete 
                        (load, domcontentloaded, networkidle)
        
        Returns:
            bool: True if navigation was successful
        """
        if not self.active_page:
            logger.error("No active page available. Start the browser first.")
            return False
        
        try:
            logger.info(f"Navigating to: {url}")
            
            # Ensure URL has a protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Navigate to the URL
            response = await self.active_page.goto(url, wait_until=wait_until, timeout=self.timeout * 1000)
            
            # Update stats
            self.session_stats["pages_visited"] += 1
            self.session_stats["last_action_time"] = datetime.now().isoformat()
            
            # Check if the navigation was successful
            if response and response.ok:
                logger.info(f"Successfully navigated to {url}")
                return True
            else:
                status = response.status if response else "unknown"
                logger.warning(f"Navigation to {url} returned status {status}")
                return False
            
        except Exception as e:
            logger.error(f"Error navigating to {url}: {str(e)}")
            self.session_stats["errors"] += 1
            return False
    
    async def get_page_content(self) -> str:
        """
        Get the HTML content of the active page
        
        Returns:
            str: The HTML content of the page
        """
        if not self.active_page:
            logger.error("No active page available. Start the browser first.")
            return ""
        
        try:
            content = await self.active_page.content()
            return content
        except Exception as e:
            logger.error(f"Error getting page content: {str(e)}")
            self.session_stats["errors"] += 1
            return ""
    
    async def take_screenshot(self, path: str = None, full_page: bool = True) -> Optional[bytes]:
        """
        Take a screenshot of the active page
        
        Args:
            path: File path to save the screenshot, if None returns the bytes
            full_page: Whether to capture the full scrollable page
        
        Returns:
            Optional[bytes]: The screenshot as bytes if path is None, otherwise None
        """
        if not self.active_page:
            logger.error("No active page available. Start the browser first.")
            return None
        
        try:
            # Generate default path if not provided
            if not path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"screenshots/screenshot_{timestamp}.png"
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Take screenshot
            screenshot = await self.active_page.screenshot(path=path, full_page=full_page)
            
            # Update stats
            self.session_stats["screenshots_taken"] += 1
            self.session_stats["last_action_time"] = datetime.now().isoformat()
            
            logger.info(f"Screenshot taken: {path}")
            return screenshot
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            self.session_stats["errors"] += 1
            return None
    
    async def click(self, selector: str, timeout: int = None) -> bool:
        """
        Click on an element identified by the selector
        
        Args:
            selector: CSS or XPath selector for the element
            timeout: Custom timeout in seconds for this operation
        
        Returns:
            bool: True if the click was successful
        """
        if not self.active_page:
            logger.error("No active page available. Start the browser first.")
            return False
        
        try:
            timeout_ms = (timeout or self.timeout) * 1000
            await self.active_page.click(selector, timeout=timeout_ms)
            
            self.session_stats["last_action_time"] = datetime.now().isoformat()
            logger.info(f"Clicked element: {selector}")
            return True
        except Exception as e:
            logger.error(f"Error clicking element {selector}: {str(e)}")
            self.session_stats["errors"] += 1
            return False
    
    async def fill_form(self, form_data: Dict[str, str], submit_selector: str = None) -> bool:
        """
        Fill a form with provided data
        
        Args:
            form_data: Dictionary mapping field selectors to values
            submit_selector: Selector for the submit button (optional)
        
        Returns:
            bool: True if the form was filled successfully
        """
        if not self.active_page:
            logger.error("No active page available. Start the browser first.")
            return False
        
        try:
            # Fill each field
            for selector, value in form_data.items():
                await self.active_page.fill(selector, value)
                logger.info(f"Filled form field {selector}")
            
            # Submit the form if a submit selector is provided
            if submit_selector:
                await self.active_page.click(submit_selector)
                logger.info(f"Submitted form using {submit_selector}")
            
            # Update stats
            self.session_stats["forms_submitted"] += 1
            self.session_stats["last_action_time"] = datetime.now().isoformat()
            
            return True
        except Exception as e:
            logger.error(f"Error filling form: {str(e)}")
            self.session_stats["errors"] += 1
            return False
    
    async def login(self, url: str, username_selector: str, password_selector: str,
                   username: str, password: str, submit_selector: str) -> bool:
        """
        Perform a login on a website
        
        Args:
            url: The login page URL
            username_selector: Selector for the username field
            password_selector: Selector for the password field
            username: Username to enter
            password: Password to enter
            submit_selector: Selector for the login button
        
        Returns:
            bool: True if login was successful
        """
        if not self.active_page:
            logger.error("No active page available. Start the browser first.")
            return False
        
        try:
            # Navigate to the login page
            await self.navigate(url)
            
            # Fill login form
            form_data = {
                username_selector: username,
                password_selector: password
            }
            success = await self.fill_form(form_data, submit_selector)
            
            if success:
                # Wait for navigation after login
                await self.active_page.wait_for_load_state("networkidle", timeout=self.timeout * 1000)
                
                # Update stats
                self.session_stats["login_attempts"] += 1
                logger.info(f"Login attempt completed on {url}")
                
                # We should add some verification of successful login here in a real implementation
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            self.session_stats["errors"] += 1
            return False
    
    async def extract_data(self, selectors: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract data from the current page using selectors
        
        Args:
            selectors: Dictionary mapping data keys to CSS selectors
        
        Returns:
            Dict[str, Any]: Extracted data with keys matching the input dictionary
        """
        if not self.active_page:
            logger.error("No active page available. Start the browser first.")
            return {}
        
        result = {}
        try:
            for key, selector in selectors.items():
                try:
                    # For text content
                    element = await self.active_page.query_selector(selector)
                    if element:
                        result[key] = await element.inner_text()
                    else:
                        result[key] = None
                except Exception as inner_e:
                    logger.warning(f"Error extracting data for {key}: {str(inner_e)}")
                    result[key] = None
            
            # Update stats
            self.session_stats["data_scraped"] += len(result)
            self.session_stats["last_action_time"] = datetime.now().isoformat()
            
            logger.info(f"Extracted {len(result)} data points from page")
            return result
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            self.session_stats["errors"] += 1
            return {}
    
    async def extract_table(self, table_selector: str) -> List[Dict[str, str]]:
        """
        Extract a table from the current page
        
        Args:
            table_selector: CSS selector for the table
        
        Returns:
            List[Dict[str, str]]: List of row data as dictionaries
        """
        if not self.active_page:
            logger.error("No active page available. Start the browser first.")
            return []
        
        try:
            # Get the table rows and headers
            header_cells = await self.active_page.query_selector_all(f"{table_selector} thead th")
            headers = []
            for header in header_cells:
                header_text = await header.inner_text()
                headers.append(header_text.strip())
            
            # If no headers found, try getting them from the first row
            if not headers:
                header_cells = await self.active_page.query_selector_all(f"{table_selector} tr:first-child th, {table_selector} tr:first-child td")
                for header in header_cells:
                    header_text = await header.inner_text()
                    headers.append(header_text.strip())
            
            # Get all rows
            rows = await self.active_page.query_selector_all(f"{table_selector} tbody tr")
            
            # If no rows found with tbody, try without tbody
            if not rows:
                rows = await self.active_page.query_selector_all(f"{table_selector} tr")
                # Skip the first row if we used it for headers
                if not header_cells:
                    rows = rows[1:]
            
            results = []
            for row in rows:
                cells = await row.query_selector_all("td")
                row_data = {}
                
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        cell_text = await cell.inner_text()
                        row_data[headers[i]] = cell_text.strip()
                    else:
                        # If more cells than headers, use index as key
                        cell_text = await cell.inner_text()
                        row_data[f"column_{i}"] = cell_text.strip()
                
                if row_data:  # Only add non-empty rows
                    results.append(row_data)
            
            # Update stats
            self.session_stats["data_scraped"] += len(results)
            self.session_stats["last_action_time"] = datetime.now().isoformat()
            
            logger.info(f"Extracted table with {len(results)} rows")
            return results
        except Exception as e:
            logger.error(f"Error extracting table: {str(e)}")
            self.session_stats["errors"] += 1
            return []
    
    async def execute_script(self, script: str) -> Any:
        """
        Execute JavaScript in the browser context
        
        Args:
            script: JavaScript code to execute
        
        Returns:
            Any: Result of the script execution
        """
        if not self.active_page:
            logger.error("No active page available. Start the browser first.")
            return None
        
        try:
            result = await self.active_page.evaluate(script)
            logger.info("Executed JavaScript in browser context")
            return result
        except Exception as e:
            logger.error(f"Error executing script: {str(e)}")
            self.session_stats["errors"] += 1
            return None
    
    async def wait_for_selector(self, selector: str, timeout: int = None, state: str = "visible") -> bool:
        """
        Wait for an element to appear on the page
        
        Args:
            selector: CSS selector for the element
            timeout: Custom timeout in seconds
            state: State to wait for (attached, detached, visible, hidden)
        
        Returns:
            bool: True if the element appeared within the timeout
        """
        if not self.active_page:
            logger.error("No active page available. Start the browser first.")
            return False
        
        try:
            timeout_ms = (timeout or self.timeout) * 1000
            await self.active_page.wait_for_selector(selector, timeout=timeout_ms, state=state)
            logger.info(f"Element found: {selector}")
            return True
        except Exception as e:
            logger.error(f"Timeout waiting for element {selector}: {str(e)}")
            self.session_stats["errors"] += 1
            return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current browser session
        
        Returns:
            Dict[str, Any]: Session statistics
        """
        # Calculate uptime if session started
        if self.session_stats["start_time"]:
            start_time = datetime.fromisoformat(self.session_stats["start_time"])
            uptime_seconds = (datetime.now() - start_time).total_seconds()
            self.session_stats["uptime_seconds"] = uptime_seconds
        
        return self.session_stats
    
    async def create_new_page(self) -> bool:
        """
        Create a new page/tab in the browser
        
        Returns:
            bool: True if the new page was created successfully
        """
        if not self.browser:
            logger.error("No browser instance. Start the browser first.")
            return False
        
        try:
            new_page = await self.browser.new_page()
            self.pages.append(new_page)
            self.active_page = new_page
            
            logger.info("Created new browser page")
            return True
        except Exception as e:
            logger.error(f"Error creating new page: {str(e)}")
            self.session_stats["errors"] += 1
            return False
    
    async def switch_to_page(self, index: int) -> bool:
        """
        Switch the active page to the one at the specified index
        
        Args:
            index: Index of the page to switch to
        
        Returns:
            bool: True if the switch was successful
        """
        if not self.browser or not self.pages:
            logger.error("No browser instance or pages. Start the browser first.")
            return False
        
        try:
            if 0 <= index < len(self.pages):
                self.active_page = self.pages[index]
                logger.info(f"Switched to page at index {index}")
                return True
            else:
                logger.error(f"Invalid page index: {index}")
                return False
        except Exception as e:
            logger.error(f"Error switching to page {index}: {str(e)}")
            self.session_stats["errors"] += 1
            return False


# For testing
async def test_browser_controller():
    """Function to test the BrowserController"""
    # Initialize the controller
    controller = BrowserController(headless=False)
    
    # Start the browser
    success = await controller.start_browser()
    if not success:
        return
    
    try:
        # Navigate to a website
        await controller.navigate("https://example.com")
        
        # Take a screenshot
        await controller.take_screenshot("example_screenshot.png")
        
        # Extract some data
        data = await controller.extract_data({
            "title": "h1",
            "paragraph": "p"
        })
        
        print("Extracted data:", data)
        
        # Get session stats
        stats = controller.get_session_stats()
        print("Session stats:", stats)
        
    finally:
        # Close the browser
        await controller.close_browser()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_browser_controller()) 