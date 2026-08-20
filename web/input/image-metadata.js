(function installImageMetadata(global) {
  function asciiAt(bytes, offset, text) {
    if (offset < 0 || offset + text.length > bytes.length) return false;
    for (let index = 0; index < text.length; index += 1) {
      if (bytes[offset + index] !== text.charCodeAt(index)) return false;
    }
    return true;
  }

  function parsedImageSize(width, height, format, orientation = 1) {
    const w = Math.round(Number(width) || 0);
    const h = Math.round(Number(height) || 0);
    if (w <= 0 || h <= 0) return null;
    const result = { width: w, height: h, format };
    const normalizedOrientation = Math.round(Number(orientation) || 1);
    if (normalizedOrientation >= 2 && normalizedOrientation <= 8) {
      result.orientation = normalizedOrientation;
    }
    return result;
  }

  function parseTiffOrientation(bytes) {
    if (bytes.length < 16) return 1;
    const little = asciiAt(bytes, 0, "II");
    const big = asciiAt(bytes, 0, "MM");
    if (!little && !big) return 1;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const uint16 = (offset) => offset + 2 <= bytes.length ? view.getUint16(offset, little) : 0;
    const uint32 = (offset) => offset + 4 <= bytes.length ? view.getUint32(offset, little) : 0;
    if (uint16(2) !== 42) return 1;
    const ifd = uint32(4);
    if (ifd <= 0 || ifd + 2 > bytes.length) return 1;
    const entries = Math.min(uint16(ifd), Math.floor((bytes.length - ifd - 2) / 12));
    for (let entry = 0; entry < entries; entry += 1) {
      const offset = ifd + 2 + entry * 12;
      if (uint16(offset) !== 0x0112 || uint16(offset + 2) !== 3 || uint32(offset + 4) !== 1) continue;
      const orientation = uint16(offset + 8);
      return orientation >= 1 && orientation <= 8 ? orientation : 1;
    }
    return 1;
  }

  function orientationSwapsImageAxes(orientation) {
    return [5, 6, 7, 8].includes(Math.round(Number(orientation) || 1));
  }

  function displayOrientedImageSize(width, height, orientation = 1) {
    return orientationSwapsImageAxes(orientation)
      ? [Math.max(1, height), Math.max(1, width)]
      : [Math.max(1, width), Math.max(1, height)];
  }

  function parseTiffDimensions(bytes) {
    if (bytes.length < 16) return null;
    const little = asciiAt(bytes, 0, "II");
    const big = asciiAt(bytes, 0, "MM");
    if (!little && !big) return null;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const uint16 = (offset) => offset + 2 <= bytes.length ? view.getUint16(offset, little) : 0;
    const uint32 = (offset) => offset + 4 <= bytes.length ? view.getUint32(offset, little) : 0;
    if (uint16(2) !== 42) return null;
    const ifd = uint32(4);
    if (ifd <= 0 || ifd + 2 > bytes.length) return null;
    const entries = Math.min(uint16(ifd), Math.floor((bytes.length - ifd - 2) / 12));
    let width = 0;
    let height = 0;
    for (let entry = 0; entry < entries; entry += 1) {
      const offset = ifd + 2 + entry * 12;
      const tag = uint16(offset);
      if (tag !== 256 && tag !== 257) continue;
      const type = uint16(offset + 2);
      const count = uint32(offset + 4);
      if (count !== 1) continue;
      const value = type === 3 ? uint16(offset + 8) : type === 4 ? uint32(offset + 8) : 0;
      if (tag === 256) width = value;
      else height = value;
      if (width && height) return parsedImageSize(width, height, "tiff");
    }
    return null;
  }

  function parseImageDimensions(bytes, mimeType = "") {
    if (bytes.length >= 24 &&
        bytes[0] === 0x89 && asciiAt(bytes, 1, "PNG\r\n\u001a\n")) {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      return parsedImageSize(view.getUint32(16), view.getUint32(20), "png");
    }
    if (bytes.length >= 10 && (asciiAt(bytes, 0, "GIF87a") || asciiAt(bytes, 0, "GIF89a"))) {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      return parsedImageSize(view.getUint16(6, true), view.getUint16(8, true), "gif");
    }
    if (bytes.length >= 30 && asciiAt(bytes, 0, "RIFF") && asciiAt(bytes, 8, "WEBP")) {
      if (asciiAt(bytes, 12, "VP8X")) {
        const width = 1 + bytes[24] + (bytes[25] << 8) + (bytes[26] << 16);
        const height = 1 + bytes[27] + (bytes[28] << 8) + (bytes[29] << 16);
        return parsedImageSize(width, height, "webp-vp8x");
      }
      if (asciiAt(bytes, 12, "VP8 ") && bytes.length >= 30 && bytes[23] === 0x9d && bytes[24] === 0x01 && bytes[25] === 0x2a) {
        const width = (bytes[26] | (bytes[27] << 8)) & 0x3fff;
        const height = (bytes[28] | (bytes[29] << 8)) & 0x3fff;
        return parsedImageSize(width, height, "webp-vp8");
      }
      if (asciiAt(bytes, 12, "VP8L") && bytes.length >= 25 && bytes[20] === 0x2f) {
        const width = 1 + bytes[21] + ((bytes[22] & 0x3f) << 8);
        const height = 1 + (bytes[22] >> 6) + (bytes[23] << 2) + ((bytes[24] & 0x0f) << 10);
        return parsedImageSize(width, height, "webp-vp8l");
      }
    }
    if (bytes.length >= 26 && asciiAt(bytes, 0, "BM")) {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      return parsedImageSize(
        Math.abs(view.getInt32(18, true)),
        Math.abs(view.getInt32(22, true)),
        "bmp",
      );
    }
    if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xd8) {
      const startOfFrame = new Set([
        0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
        0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
      ]);
      const segmentsWithoutPayload = new Set([0x00, 0x01, 0xd8, 0xd9]);
      for (let marker = 0xd0; marker <= 0xd7; marker += 1) {
        segmentsWithoutPayload.add(marker);
      }
      let offset = 2;
      let orientation = 1;
      const maxOffset = bytes.length - 1;
      while (offset < maxOffset) {
        if (bytes[offset] !== 0xff) {
          offset += 1;
          continue;
        }
        while (offset < maxOffset && bytes[offset] === 0xff) {
          offset += 1;
        }
        if (offset >= maxOffset) break;

        const markerCode = bytes[offset];
        if (markerCode === 0xda) return null;
        if (startOfFrame.has(markerCode)) {
          if (offset + 7 >= bytes.length) return null;
          const frameLength = (bytes[offset + 1] << 8) | bytes[offset + 2];
          if (frameLength < 8 || offset + 1 + frameLength > bytes.length) return null;
          const height = (bytes[offset + 4] << 8) | bytes[offset + 5];
          const width = (bytes[offset + 6] << 8) | bytes[offset + 7];
          return parsedImageSize(width, height, "jpeg", orientation);
        }

        if (segmentsWithoutPayload.has(markerCode)) {
          offset += 1;
          continue;
        }

        if (offset + 2 >= bytes.length) return null;
        const segmentLength = (bytes[offset + 1] << 8) | bytes[offset + 2];
        if (segmentLength < 2) return null;
        const nextOffset = offset + 1 + segmentLength;
        if (nextOffset > bytes.length) return null;
        const payloadOffset = offset + 3;
        if (markerCode === 0xe1 && asciiAt(bytes, payloadOffset, "Exif\0\0")) {
          orientation = parseTiffOrientation(bytes.subarray(payloadOffset + 6, nextOffset));
        }
        offset = nextOffset;
      }
    }
    const tiff = parseTiffDimensions(bytes);
    if (tiff) return tiff;
    if (/avif|heic|heif/i.test(mimeType) || asciiAt(bytes, 4, "ftyp")) {
      for (let offset = 4; offset + 16 <= bytes.length; offset += 1) {
        if (!asciiAt(bytes, offset, "ispe")) continue;
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const width = view.getUint32(offset + 8);
        const height = view.getUint32(offset + 12);
        const parsed = parsedImageSize(width, height, "isobmff-ispe");
        if (parsed) return parsed;
      }
    }
    return null;
  }

  global.Image2SplatPaintImageMetadata = Object.freeze({
    displayOrientedImageSize,
    orientationSwapsImageAxes,
    parseImageDimensions,
  });
})(globalThis);
