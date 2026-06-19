// Theme and styling utilities
import { CSSProperties } from 'react';

// Color scheme based on mode
export const getColors = (isDarkMode: boolean) => ({
    background: isDarkMode ? '#0a0a0a' : '#ffffff',
    text: isDarkMode ? '#ffffff' : '#000000',
    accent: '#d4a574', // Slightly darker beige for better contrast
    buttonBg: isDarkMode ? '#ffffff' : '#000000',
    buttonText: isDarkMode ? '#000000' : '#ffffff',
    border: isDarkMode ? '#d1d5db' : '#374151'
});

// Common button styles
export const getButtonStyles = (isDarkMode: boolean, colors = getColors(isDarkMode)) => ({
    primary: {
        borderRadius: '9999px',
        border: `1px solid ${isDarkMode ? colors.buttonBg : colors.text}`,
        background: isDarkMode ? colors.buttonBg : 'transparent',
        color: isDarkMode ? colors.buttonText : colors.text,
        fontWeight: '600',
        fontSize: '1.125rem',
        height: '48px',
        padding: '0 20px',
        cursor: 'pointer',
        transition: 'all 0.3s ease'
    } as CSSProperties,
    secondary: {
        borderRadius: '9999px',
        border: `1px solid ${colors.border}`,
        background: 'transparent',
        color: colors.text,
        fontWeight: '500',
        fontSize: '14px',
        height: '40px',
        padding: '0 20px',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
    } as CSSProperties,
    menu: {
        fontSize: '18px',
        fontWeight: '500',
        color: colors.text,
        background: 'transparent',
        border: 'none',
        cursor: 'pointer',
        transition: 'color 0.3s ease',
        padding: '8px'
    } as CSSProperties
});

// Input field styles
export const getInputStyles = (hasError: boolean, colors = getColors(true)) => ({
    borderBottom: `2px solid ${hasError ? '#ef4444' : colors.accent}`,
    backgroundColor: 'transparent',
    color: colors.text,
    width: '100%',
    padding: '12px 4px',
    outline: 'none',
    transition: 'border-color 0.3s ease'
} as CSSProperties);

// Common layout styles
export const layoutStyles = {
    fullSection: {
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        scrollSnapAlign: 'start',
        position: 'relative',
    } as CSSProperties,
    contentContainer: {
        display: 'grid',
        alignItems: 'center',
        justifyItems: 'center',
        padding: '8vh 3.2vw 8vh 3.2vw',
        gap: '6.4vh',
        maxWidth: '1200px',
        margin: '0 auto'
    } as CSSProperties,
    flexColumn: {
        display: 'flex',
        flexDirection: 'column' as const,
        gap: '3.2vh',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
        padding: '0 1.6vw'
    } as CSSProperties
};

// Dark/Light mode toggle styles
export const getToggleStyles = (isDarkMode: boolean) => ({
    container: {
        position: 'fixed' as const,
        bottom: '15px',
        right: '15px',
        zIndex: 1500
    } as CSSProperties,
    button: {
        width: '32px',
        height: '16px',
        borderRadius: '8px',
        border: 'none',
        background: getColors(isDarkMode).accent,
        cursor: 'pointer',
        position: 'relative' as const,
        transition: 'background 0.3s ease',
        outline: 'none'
    } as CSSProperties,
    circle: {
        width: '12px',
        height: '12px',
        borderRadius: '50%',
        background: isDarkMode ? '#000000' : '#ffffff',
        position: 'absolute' as const,
        top: '2px',
        left: isDarkMode ? '2px' : '18px',
        transition: 'left 0.3s ease, background-color 0.3s ease',
        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.3)'
    } as CSSProperties
}); 