interface RotatingTextProps {
    isDarkMode?: boolean;
}

const PHRASES = [
    'your dream house',
    'your next vehicle',
    'a perfect getaway',
    'a grand retirement',
    'financial freedom',
];

const RotatingText: React.FC<RotatingTextProps> = ({ isDarkMode = true }) => {
    const textColor = '#deb887'; // Keep accent color same in both modes

    return (
        <div className="rotating-text-container">
            <div className="static-text" style={{ color: isDarkMode ? '#ffffff' : '#000000' }}>
                Start planning for
            </div>
            <div className="rotating-text-wrapper">
                {PHRASES.map((phrase, i) => (
                    <span key={i} className="rotating-text" style={{ color: textColor }}>{phrase}</span>
                ))}
                {/* Invisible sizer — kept LAST so it never offsets the
                    .rotating-text:nth-child(...) animation delays. Because the
                    rotating spans are position:absolute they give the wrapper no
                    width; this in-flow copy of every phrase makes the wrapper as
                    wide as the WIDEST phrase in whatever font is rendered, so none
                    clip — with no hard-coded width. */}
                <span className="rotating-text-sizer" aria-hidden="true">
                    {PHRASES.map((phrase, i) => (
                        <span key={i}>{phrase}</span>
                    ))}
                </span>
            </div>
        </div>
    );
};

export default RotatingText;
